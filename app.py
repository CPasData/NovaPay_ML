import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# ============================================================
# INICIALIZACIÓN
# ============================================================
app = FastAPI(
    title="NovaPay ML API",
    description="API de detección de fraude para NovaPay",
    version="3.0.0"
)

# ============================================================
# CARGA DEL MODELO AL ARRANCAR
# Un solo pkl con todo dentro:
# fe, scaler, imputer, lgb_model, xgb_model, best_w, best_t
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "modelo_07_v1.pkl")
modelo     = joblib.load(MODEL_PATH)

fe        = modelo['fe']         # FeatureEngineer
scaler    = modelo['scaler']     # StandardScaler
imputer   = modelo['imputer']    # KNNImputer
lgb       = modelo['lgb_model']  # LightGBM
xgb_m     = modelo['xgb_model']  # XGBoost
best_w    = modelo['best_w']     # peso del ensemble (LGB=30%, XGB=70%)
best_t    = modelo['best_t']     # threshold de decisión (0.314)
num_feats = modelo['num_feats']  # features numéricas para scaler e imputer

print(f"Modelo cargado correctamente")
print(f"Threshold: {best_t:.4f} | Peso LGB: {best_w} | Peso XGB: {1-best_w}")

# ============================================================
# CONFIGURACIÓN BASE DE DATOS
# ============================================================
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "database": "novapay",
    "user"    : "postgres",
    "password": "123456"  # cambia esto por tu contraseña
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# ============================================================
# MODELO DE DATOS — lo que recibe la API
# Usa nombres largos del CSV
# ============================================================
class Transaccion(BaseModel):

    # Identificación
    id_transaccion: str = Field(..., json_schema_extra={"example": "059638c5-40f"})

    # Datos del cliente
    id_cliente:                     str   = Field(..., json_schema_extra={"example": "3ddebd45-ccd"})
    tipo_cliente:                   str   = Field(..., json_schema_extra={"example": "persona"})
    edad_cliente:                   int   = Field(..., json_schema_extra={"example": 35})
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
    fecha_hora:                     str   = Field(..., json_schema_extra={"example": "2026-05-23 14:30:00"})
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
# 1. FeatureEngineer → calcula 67 features
# 2. Extraer campos calculados ANTES de imputer y scaler
# 3. Imputer → imputa nulos en las 67 features
# 4. Scaler  → escala las features numéricas
# 5. Ensemble LGB + XGB → predice probabilidad
# 6. Threshold 0.314 → decides is_fraud
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

    # PASO 3 — Imputer
    X[num_feats] = imputer.transform(X[num_feats])

    # PASO 4 — Scaler
    X[num_feats] = scaler.transform(X[num_feats])

    # PASO 5 — Ensemble predict
    prob_lgb = lgb.predict_proba(X)[:, 1]
    prob_xgb = xgb_m.predict_proba(X)[:, 1]
    prob     = best_w * prob_lgb + (1 - best_w) * prob_xgb

    # PASO 6 — Threshold
    is_fraud       = int((prob[0] >= best_t))
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


def guardar_en_bd(transaccion: Transaccion, prediccion: Prediccion):
    """Guarda la transacción con su predicción en PostgreSQL."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO transacciones (
                id_transaccion, id_cliente,
                id_cuenta, cuenta_origen, estado_cuenta,
                saldo_actual, saldo_avg_30d,
                vol_entrante_30d, vol_saliente_30d,
                num_trans_recib_7d, num_trans_env_7d,
                id_tarjeta, estado_tarjeta,
                fecha_creacion_tarjeta, tiempo_activo_tarjeta,
                limite_impot_tx, veces_superar_limite_7d,
                tipo_transaccion, fecha_trans,
                is_night, is_weekend,
                tiempo_desde_ult_trans, num_trans_ult_hora,
                importe_transaccion, metod_auten,
                num_veces_pin, ident_disp,
                dispositivo_reconocido,
                operacion_pais, operacion_region,
                dir_ip_origen, geolocalizacion,
                cuenta_destino, destino_alto_riesgo,
                is_fraud, prob_fraud, impacto_fraude,
                es_transfronteriza, ratio_imp_limite,
                intensidad_tx, severidad_tx, flujo_neto_30d,
                estado_revision
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s
            )
            ON CONFLICT (id_transaccion) DO NOTHING
        """, (
            transaccion.id_transaccion, transaccion.id_cliente,
            transaccion.id_cuenta, transaccion.cuenta_origen,
            transaccion.estado_cuenta, transaccion.saldo_actual,
            transaccion.saldo_medio_30_dias,
            transaccion.volumen_entrante_30_dias,
            transaccion.volumen_saliente_30_dias,
            transaccion.numero_transferencias_recibidas_7_dias,
            transaccion.numero_transferencias_enviadas_7_dias,
            transaccion.id_tarjeta, transaccion.estado_tarjeta,
            transaccion.fecha_creacion_tarjeta,
            transaccion.antiguedad_tarjeta_dias,
            transaccion.limite_importe_transacciones,
            transaccion.veces_superar_limite_7_dias,
            transaccion.tipo_transaccion, transaccion.fecha_hora,
            transaccion.is_night, transaccion.is_weekend,
            transaccion.tiempo_desde_ultima_transaccion,
            transaccion.numero_transacciones_ultima_hora,
            transaccion.importe_transaccion,
            transaccion.metodo_autenticacion,
            transaccion.numero_pin_disponibles,
            transaccion.identificador_dispositivo_fingerprint,
            transaccion.dispositivo_reconocido,
            transaccion.operacion_pais, transaccion.operacion_region,
            transaccion.direccion_ip_origen,
            transaccion.geolocalizacion, transaccion.cuenta_destino,
            transaccion.destino_alto_riesgo,
            prediccion.is_fraud, prediccion.prob_fraud,
            prediccion.impacto_fraude,
            prediccion.es_transfronteriza,
            prediccion.ratio_imp_limite,
            prediccion.intensidad_tx, prediccion.severidad_tx,
            prediccion.flujo_neto_30d,
            'pendiente' if prediccion.is_fraud == 1 else 'legitima'
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error al guardar en BD: {e}")


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", tags=["Sistema"])
def health():
    """
    Comprueba que la API esta funcionando.
    Este endpoint no espera datos, solo se llama.
    """
    return {
        "status"   : "ok",
        "version"  : "3.0.0",
        "modelo"   : "modelo_07_v1",
        "ensemble" : f"LGB {best_w:.0%} + XGB {1-best_w:.0%}",
        "threshold": round(best_t, 4),
        "metadata" : modelo.get('metadata', {})
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
        guardar_en_bd(transaccion, prediccion)
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
            guardar_en_bd(transaccion, prediccion)
            resultados.append(prediccion)

        return {
            "total"       : len(resultados),
            "fraudes"     : sum(1 for r in resultados if r.is_fraud == 1),
            "legitimas"   : sum(1 for r in resultados if r.is_fraud == 0),
            "predicciones": resultados
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Modelo ML"])
def metrics():
    """
    Devuelve metricas del modelo calculadas desde la BD.
    Este endpoint no espera datos, solo se llama.
    """
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                COUNT(*)                                        AS total_predicciones,
                SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END)  AS fraudes_detectados,
                SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END)  AS legitimas,
                ROUND(AVG(prob_fraud)::numeric, 4)              AS prob_fraud_media,
                SUM(CASE WHEN estado_revision = 'pendiente'
                     THEN 1 ELSE 0 END)                        AS pendientes_revision,
                SUM(CASE WHEN estado_revision = 'confirmado_fraude'
                     THEN 1 ELSE 0 END)                        AS confirmados_fraude,
                SUM(CASE WHEN estado_revision = 'falso_positivo'
                     THEN 1 ELSE 0 END)                        AS falsos_positivos
            FROM transacciones
            WHERE is_fraud IS NOT NULL
        """)
        row     = cur.fetchone()
        cur.close()
        conn.close()
        total   = row['total_predicciones'] or 0
        fraudes = row['fraudes_detectados'] or 0
        return {
            "total_predicciones" : total,
            "fraudes_detectados" : fraudes,
            "legitimas"          : row['legitimas'] or 0,
            "tasa_deteccion"     : round(fraudes / total, 4) if total > 0 else 0,
            "prob_fraud_media"   : float(row['prob_fraud_media'] or 0),
            "pendientes_revision": row['pendientes_revision'] or 0,
            "confirmados_fraude" : row['confirmados_fraude'] or 0,
            "falsos_positivos"   : row['falsos_positivos'] or 0,
        }
    except Exception as e:
        return {"error": str(e), "nota": "Verifica la conexion a la BD"}


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
