from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ============================================================
# INICIALIZACIÓN
# ============================================================
app = FastAPI(
    title="NovaPay ML API",
    description="API de detección de fraude - NovaPay",
    version="1.0.0"
)

# ============================================================
# MODELO DE DATOS - lo que recibe el endpoint /predict
# Basado en la vista vw_Transacciones_ML
# ============================================================
class Transaccion(BaseModel):  # Class Transaccion hereda capacidades de validacion de la clase BaseModel

    # Identificación
    id_transaccion: str = Field(..., example="059638c5-40f")
    #Field()--> añade información extra
    #...  --> el campo no debe faltar

    # Datos del cliente
    tipo_cliente:                   str     = Field(..., example="persona")
    edad_cliente:                   int     = Field(..., example=35)
    customer_country:               str     = Field(..., example="ES")
    customer_region:                str     = Field(..., example="Centro")
    tenure:                         int     = Field(..., example=365)
    importe_medio_mensual:          float   = Field(..., example=500.00)
    desviacion_estandar_mensual:    float   = Field(..., example=150.00)
    media_transacciones_al_dia:     float   = Field(..., example=3.5)
    numero_fraudes_ultimo_ano:      int     = Field(..., example=0)

    # Datos de la cuenta
    estado_cuenta:                              str     = Field(..., example="activa")
    saldo_actual:                               float   = Field(..., example=2500.00)
    saldo_medio_30_dias:                        float   = Field(..., example=2200.00)
    volumen_entrante_30_dias:                   float   = Field(..., example=3000.00)
    volumen_saliente_30_dias:                   float   = Field(..., example=2800.00)
    numero_transferencias_recibidas_7_dias:     int     = Field(..., example=3)
    numero_transferencias_enviadas_7_dias:      int     = Field(..., example=2)

    # Datos de la tarjeta
    estado_tarjeta:                 str     = Field(..., example="activa")
    antiguedad_tarjeta_dias:        int     = Field(..., example=365)
    limite_importe_transacciones:   float   = Field(..., example=2000.00)
    veces_superar_limite_7_dias:    int     = Field(..., example=0)

    # Datos de la transacción
    tipo_transaccion:                       str     = Field(..., example="tarjeta")
    is_night:                               int     = Field(..., example=0)
    is_weekend:                             int     = Field(..., example=0)
    tiempo_desde_ultima_transaccion:        int     = Field(..., example=3600)
    numero_transacciones_ultima_hora:       int     = Field(..., example=1)
    importe_transaccion:                    float   = Field(..., example=150.00)
    metodo_autenticacion:                   str     = Field(..., example="PIN")
    numero_pin_disponibles:                 int     = Field(..., example=3)
    dispositivo_reconocido:                 int     = Field(..., example=1)
    operacion_pais:                         str     = Field(..., example="ES")
    operacion_region:                       str     = Field(..., example="Centro")
    destino_alto_riesgo:                    int     = Field(..., example=0)


# ============================================================
# MODELO DE RESPUESTA — lo que devuelve el endpoint /predict
# ============================================================
class Prediccion(BaseModel):
    id_transaccion:     str
    is_fraud:           int     # 0 = legítima, 1 = fraude
    prob_fraud:         float   # probabilidad 0.0 a 1.0
    impacto_fraude:     int     # 0=no fraude, 1=bajo, 2=medio, 3=alto
    mensaje:            str


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
# ENDPOINTS
# ============================================================

# --- HEALTH CHECK ---
@app.get("/health", tags=["Sistema"])
def health():
    """
    Comprueba que la API está funcionando.
    """
    # Este endpoint no espera datos
    # Solo se llama: GET http://localhost:8000/health
    return {
        "status": "ok",
        "version": "1.0.0",
        "modelo": "simulado — pendiente de modelo real"  # ****** para modificar cuando se tenga modelo
    }


# --- PREDICT ---
# response_model=Prediccion --> lo que devuelve la API tiene la forma de la clase Prediccion
@app.post("/predict", response_model=Prediccion, tags=["Modelo ML"])
def predict(transaccion: Transaccion):   #FastAPI valida que la que los datos que llegan tiene la forma de la clase Prediccion Transaccion
    """
    Recibe una transacción y devuelve si es fraude o no.

    - Ciber usa este endpoint para atacar con transacciones sospechosas
    - Full Stack llama a este endpoint cuando entra una transacción nueva
    - Devuelve is_fraud (0/1), prob_fraud (0.0-1.0) e impacto_fraude (0-3) (bajo,medio, alto)
    """

    # ----------------------------------------------------------------
    # MODELO SIMULADO
    # Mientras el modelo real no está listo devuelve
    # una predicción basada en reglas simples
    # Cuando llegue el modelo real solo hay que reemplazar
    # ----------------------------------------------------------------

    prob = 0.01  # probabilidad base

    # Reglas simples de detección
    if transaccion.customer_country != transaccion.operacion_pais:
        prob += 0.15
    if transaccion.dispositivo_reconocido == 0:
        prob += 0.10
    if transaccion.estado_cuenta == "bloqueada":
        prob += 0.15
    if transaccion.estado_tarjeta in ("robada", "extraviada", "bloqueada"):
        prob += 0.25
    if transaccion.importe_transaccion > transaccion.limite_importe_transacciones * 0.9:
        prob += 0.10
    if transaccion.is_night and transaccion.importe_transaccion > 500:
        prob += 0.05
    if transaccion.numero_transacciones_ultima_hora > 5:
        prob += 0.08
    if transaccion.tiempo_desde_ultima_transaccion < 60 and transaccion.importe_transaccion > 1000:
        prob += 0.06
    if transaccion.veces_superar_limite_7_dias > 3:
        prob += 0.10
    if transaccion.numero_pin_disponibles == 0:
        prob += 0.08
    if transaccion.destino_alto_riesgo == 1:
        prob += 0.20
    if transaccion.numero_fraudes_ultimo_ano > 0:
        prob += 0.10

    prob = min(prob, 0.95)  # máximo 95%

    is_fraud = 1 if prob >= 0.5 else 0
    impacto = calcular_impacto(is_fraud, transaccion.importe_transaccion)

    return Prediccion(
        id_transaccion  = transaccion.id_transaccion,
        is_fraud        = is_fraud,
        prob_fraud      = round(prob, 4),
        impacto_fraude  = impacto,
        mensaje         = "fraude detectado" if is_fraud == 1 else "transacción legítima"
    )


# --- METRICS ---
# 
@app.get("/metrics", tags=["Modelo ML"])
def metrics():
    """
    Devuelve las métricas del modelo.
    Útil para el dashboard de KPIs.
    """
    # Este endpoint no espera datos
    # Solo se llama: GET http://localhost:8000/metrics
    return {
        "modelo_activo":    "simulado",
        "version":          "ronda_1",
        "total_predicciones": 0,
        "fraudes_detectados": 0,
        "tasa_deteccion":   0.0,
        "nota": "métricas reales disponibles cuando se integre el modelo .pkl"  # a espera del modelo esto se modificara
    }


# ============================================================
# ARRANCAR LA API
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # recarga automática al guardar cambios
    )
