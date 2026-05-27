import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
import uvicorn
import joblib
import pandas as pd

# ============================================================
# INICIALIZACIÓN
# ============================================================
app = FastAPI(
    title="NovaPay ML API",
    description="API de detección de fraude para NovaPay",
    version="5.0.0"
)

# ============================================================
# CARGA DEL MODELO AL ARRANCAR
# Un solo pkl con todo dentro:
# fe, scaler, imputer, lgb_model, xgb_model, best_w, best_t
# ============================================================
MODEL_NAME = "modelo_09_v3.pkl"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", MODEL_NAME)
modelo     = joblib.load(MODEL_PATH)

fe        = modelo['fe']         # FeatureEngineer
scaler    = modelo['scaler']     # StandardScaler
imputer   = modelo['imputer']    # KNNImputer
lgb       = modelo['lgb_model']  # LightGBM
xgb_m     = modelo['xgb_model']  # XGBoost
best_w    = modelo['best_w']     # peso del ensemble
best_t    = modelo['best_t']     # threshold de decisión global
num_feats = modelo['num_feats']  # features numéricas para scaler e imputer

per_channel_thr = modelo.get('per_channel_thresholds', {})
metadata_extra  = modelo.get('metadata', {})

print(f"Modelo cargado: {MODEL_NAME}")
print(f"Threshold: {best_t:.4f} | Peso LGB: {best_w} | Peso XGB: {1-best_w}")
if per_channel_thr:
    print(f"Thresholds por canal: {per_channel_thr}")

# ============================================================
# MODELO DE DATOS — lo que recibe la API
# Usa nombres largos del CSV
# ============================================================
class Transaccion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1)

    # Identificación
    id_transaccion: str = Field(..., json_schema_extra={"example": "059638c5-40f"})
    #Field()--> añade información extra
    #...  --> el campo no debe faltar

    # Datos del cliente
    id_cliente:                     str   = Field(..., json_schema_extra={"example": "3ddebd45-ccd"})
    tipo_cliente:                   str   = Field(..., json_schema_extra={"example": "persona"})
    edad_cliente:                   int   = Field(..., ge=0, le=120, json_schema_extra={"example": 35})
    customer_country:               str   = Field(..., json_schema_extra={"example": "ES"})
    customer_region:                str   = Field(..., json_schema_extra={"example": "Centro"})
    tenure:                         int   = Field(..., json_schema_extra={"example": 365})
    importe_medio_mensual:          float = Field(..., json_schema_extra={"example": 500.00})
    desviacion_estandar_mensual:    float = Field(..., json_schema_extra={"example": 150.00})
    media_transacciones_al_dia:     float = Field(..., json_schema_extra={"example": 3.5})
    numero_fraudes_ultimo_ano:      int   = Field(..., json_schema_extra={"example": 0})

    # Datos de cuenta
    id_cuenta:                              str   = Field(..., json_schema_extra={"example": "8f3262c1-69a"})
    cuenta_origen:                          str   = Field(..., json_schema_extra={"example": "ES20427866183"})
    estado_cuenta:                          str   = Field(..., json_schema_extra={"example": "activa"})
    saldo_actual:                           float = Field(..., json_schema_extra={"example": 2500.00})
    saldo_medio_30_dias:                    float = Field(..., json_schema_extra={"example": 2200.00})
    volumen_entrante_30_dias:               float = Field(..., json_schema_extra={"example": 3000.00})
    volumen_saliente_30_dias:               float = Field(..., json_schema_extra={"example": 2800.00})
    numero_transferencias_recibidas_7_dias: int   = Field(..., json_schema_extra={"example": 3})
    numero_transferencias_enviadas_7_dias:  int   = Field(..., json_schema_extra={"example": 2})

    # Datos de tarjeta
    id_tarjeta:                     str   = Field(..., json_schema_extra={"example": "00df680c-e19"})
    estado_tarjeta:                 str   = Field(..., json_schema_extra={"example": "activa"})
    fecha_creacion_tarjeta:         str   = Field(..., json_schema_extra={"example": "2023-01-15"})
    antiguedad_tarjeta_dias:        int   = Field(..., json_schema_extra={"example": 365})
    limite_importe_transacciones:   float = Field(..., json_schema_extra={"example": 2000.00})
    veces_superar_limite_7_dias:    int   = Field(..., json_schema_extra={"example": 0})

    # Datos de transacción
    tipo_transaccion:               str   = Field(..., json_schema_extra={"example": "tarjeta"})
    fecha_hora:                     str   = Field(..., json_schema_extra={"example": "2026-05-23T14:30:00"})
    is_night:                       int   = Field(..., json_schema_extra={"example": 0})
    is_weekend:                     int   = Field(..., json_schema_extra={"example": 0})
    tiempo_desde_ultima_transaccion:    int   = Field(..., json_schema_extra={"example": 3600})
    numero_transacciones_ultima_hora:   int   = Field(..., json_schema_extra={"example": 1})
    importe_transaccion:            float = Field(..., json_schema_extra={"example": 150.00})
    metodo_autenticacion:           str   = Field(..., json_schema_extra={"example": "PIN"})
    numero_pin_disponibles:         int   = Field(..., json_schema_extra={"example": 3})
    identificador_dispositivo_fingerprint: Optional[str] = Field(None, json_schema_extra={"example": "fa1bdf50-1c95"})
    dispositivo_reconocido:         int   = Field(..., json_schema_extra={"example": 1})
    operacion_pais:                 str   = Field(..., json_schema_extra={"example": "ES"})
    operacion_region:               str   = Field(..., json_schema_extra={"example": "Centro"})
    direccion_ip_origen:            Optional[str] = Field(None, json_schema_extra={"example": "86.34.12.179"})
    geolocalizacion:                Optional[str] = Field(None, json_schema_extra={"example": "40.4168,-3.7038"})
    cuenta_destino:                 Optional[str] = Field(None, json_schema_extra={"example": "ES169540317577"})
    destino_alto_riesgo:            int   = Field(..., json_schema_extra={"example": 0})

    @field_validator('importe_transaccion')
    @classmethod
    def importe_positivo(cls, v):
        if v <= 0:
            raise ValueError('El importe debe ser positivo')
        return v

    @field_validator('dispositivo_reconocido', 'is_night', 'is_weekend', 'destino_alto_riesgo')
    @classmethod
    def binario(cls, v):
        if v not in (0, 1):
            raise ValueError(f'Debe ser 0 o 1, se recibió {v}')
        return v

    @field_validator('numero_pin_disponibles')
    @classmethod
    def pin_no_negativo(cls, v):
        if v < 0:
            raise ValueError('numero_pin_disponibles no puede ser negativo')
        return v


# ============================================================
# MODELO DE RESPUESTA
# ============================================================
class Prediccion(BaseModel):
    id_transaccion:     str
    is_fraud:           int    # 0 = legítima, 1 = fraude
    prob_fraud:         float  # probabilidad de fraude 0.0 a 1.0
    impacto_fraude:     int    # 0=no fraude, 1=bajo, 2=medio, 3=alto
    es_transfronteriza: int    # 0 = mismo país, 1 = país diferente
    ratio_imp_limite:   float  # importe / límite tarjeta
    intensidad_tx:      float  # num_transacciones / tiempo → alto = sospechoso
    severidad_tx:       float  # importe * num_transacciones → alto = sospechoso
    flujo_neto_30d:     float  # vol_entrante - vol_saliente → negativo = sale más de lo que entra
    mensaje:            str    # "FRAUDE DETECTADO" o "Transacción legítima"

    @field_validator('prob_fraud')
    @classmethod
    def prob_entre_0_y_1(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('prob_fraud debe estar entre 0.0 y 1.0')
        return v


# ============================================================
# FUNCIÓN AUXILIAR — calcula impacto según importe
# ============================================================
def calcular_impacto(is_fraud: int, importe: float) -> int:
    if is_fraud == 0:
        return 0
    elif importe < 500:
        return 1  # bajo
    elif importe < 2000:
        return 2  # medio
    else:
        return 3  # alto

# ============================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# Orden del pipeline:
# 1. FeatureEngineer → calcula 69 features (v4)
# 2. Extraer campos calculados ANTES de imputer y scaler
# 3. Scaler  → escala las features numéricas
# 4. Imputer → imputa nulos en las 69 features
# 5. Ensemble LGB + XGB → predice probabilidad
# 6. Threshold por canal (si existe) o global → decide is_fraud
# ============================================================
def predecir(transaccion: Transaccion) -> Prediccion:
    datos  = transaccion.model_dump()
    df_row = pd.DataFrame([datos])

    # PASO 1 — Feature Engineering
    X = fe.transform(df_row)

    # PASO 2 — Extraer campos calculados ANTES de imputer y scaler
    cross_border     = int(X['cross_border'].values[0])
    txn_vs_limit_pct = float(round(X['txn_vs_limit_pct'].values[0], 4))
    txn_intensity    = float(round(X['txn_intensity'].values[0], 4))
    txn_severity     = float(round(X['txn_severity'].values[0], 4))
    net_flow_30d     = float(round(X['net_flow_30d'].values[0], 4))

    # PASO 3 — Scaler
    X[num_feats] = scaler.transform(X[num_feats])

    # PASO 4 — Imputer
    X[num_feats] = imputer.transform(X[num_feats])

    # PASO 5 — Ensemble predict
    prob_lgb = lgb.predict_proba(X)[:, 1]
    prob_xgb = xgb_m.predict_proba(X)[:, 1]
    prob     = best_w * prob_lgb + (1 - best_w) * prob_xgb

    # PASO 6 — Threshold (por canal si está disponible, sino global)
    canal = transaccion.tipo_transaccion
    thr   = per_channel_thr.get(canal, best_t)
    is_fraud       = int((prob[0] >= thr))
    prob_fraud     = float(round(prob[0], 4))
    impacto_fraude = calcular_impacto(is_fraud, transaccion.importe_transaccion)

    mensaje = (
        f"FRAUDE DETECTADO - probabilidad {prob_fraud:.0%}"
        if is_fraud == 1
        else f"Transaccion legitima - probabilidad fraude {prob_fraud:.0%}"
    )

    return Prediccion(
        id_transaccion     = transaccion.id_transaccion,
        is_fraud           = is_fraud,
        prob_fraud         = prob_fraud,
        impacto_fraude     = impacto_fraude,
        es_transfronteriza = cross_border,
        ratio_imp_limite   = txn_vs_limit_pct,
        intensidad_tx      = txn_intensity,
        severidad_tx       = txn_severity,
        flujo_neto_30d     = net_flow_30d,
        mensaje            = mensaje
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", tags=["Sistema"])
def health():
    return {
        "status"   : "ok",
        "version"  : "5.0.0",
        "modelo"   : MODEL_NAME,
        "ensemble" : f"LGB {best_w:.0%} + XGB {1-best_w:.0%}",
        "threshold_global": round(best_t, 4),
        "thresholds_canal": per_channel_thr,
        "recall_at_k"     : metadata_extra.get("recall_at_k"),
    }


@app.post("/predict", response_model=Prediccion, tags=["Modelo ML"])
def predict(transaccion: Transaccion):
    """
    Recibe UNA transaccion y devuelve la prediccion.
    Tambien guarda el resultado en PostgreSQL.
    Ciber usa este endpoint para atacar con transacciones sospechosas.
    """
    try:
        prediccion = predecir(transaccion)
        #guardar_en_bd(transaccion, prediccion)
        return prediccion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Modelo ML"])
def predict_batch(transacciones: List[Transaccion]):
    """
    Recibe MUCHAS transacciones y devuelve predicciones para todas.
    Util para que Ciber pueda atacar con muchas transacciones a la vez.
    Tambien guarda todos los resultados en PostgreSQL.
    """
    try:
        resultados = []
        for transaccion in transacciones:
            prediccion = predecir(transaccion)
            #guardar_en_bd(transaccion, prediccion)
            resultados.append(prediccion)

        return {
            "total"       : len(resultados),
            "fraudes"     : sum(1 for r in resultados if r.is_fraud == 1),
            "legitimas"   : sum(1 for r in resultados if r.is_fraud == 0),
            "predicciones": resultados
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ARRANCAR LA API
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
