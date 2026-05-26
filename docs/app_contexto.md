# NovaPay ML API — Contexto y Documentación
**Proyecto:** Desafío de Tripulaciones — Operación Centinela  
**Equipo Data Science:** Blue Team  
**Versión API:** 4.0.0  
**Fecha:** Mayo 2026

---

## ¿Qué hace la API?

La API recibe transacciones bancarias, las analiza con un modelo de Machine Learning y decide si son fraudulentas o no. Guarda el resultado automáticamente en PostgreSQL (Supabase) para que el analista pueda revisarlas desde la app de Full Stack.

---

## Archivos principales

### `app.py`
Es el corazón del proyecto de Data Science. Contiene:
- Carga del modelo ML al arrancar la API
- 4 endpoints para recibir transacciones y devolver predicciones:
  - `GET  /health`         → Comprueba que la API está activa
  - `POST /predict`        → Predice UNA transacción y guarda en la base de datos  PostgreSQL
  - `POST /predict/batch`  → Predice MUCHAS transacciones a la vez y guarda en la base de datos  PostgreSQL
  - `GET  /metrics`        → Devuelve métricas del modelo desde la BD en tiempo real
- Conexión a PostgreSQL para guardar resultados automáticamente
- Pipeline de predicción que aplica el modelo paso a paso

### `test_api.ipynb`
Notebook de pruebas para verificar que la API funciona correctamente en local. Contiene pruebas para los 4 endpoints:
- `GET /health` → Verifica que la API está activa y el modelo cargado correctamente
- `POST /predict` → Prueba con 1 transacción legítima y 1 transacción sospechosa (simulando ataque de Ciber)
- `POST /predict/batch` → Prueba batch de 10 filas del CSV con comparativa vs valores reales del CSV
- `GET /metrics` → Verifica que las métricas se calculan correctamente desde la BD

---

## ¿Cómo arrancar la API?

```bash
# Requisitos previos:
# → Docker corriendo con postgres-demo
# → BD novapay creada con las tablas

python app.py
```

La API estará disponible en:
- **URL:** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`

---

## El modelo ML

El modelo está guardado en un único archivo `.pkl` que contiene todo:

```
modelo_07_v1.pkl
├── fe         → FeatureEngineer (genera 67 features automáticamente)
├── imputer    → KNNImputer (rellena valores nulos)
├── scaler     → StandardScaler (escala los datos)
├── lgb_model  → LightGBM
├── xgb_model  → XGBoost
├── best_w     → peso ensemble (LGB 30% + XGB 70%)
└── best_t     → threshold de decisión (0.3144)
```

El pipeline de predicción sigue este orden:

```
Transacción entra a la API
        ↓
1. FeatureEngineer  →  genera 67 features automáticamente
        ↓
2. Extrae campos calculados  →  es_transfronteriza, intensidad_tx...
        ↓
3. KNNImputer  →  rellena valores nulos
        ↓
4. StandardScaler  →  escala los números
        ↓
5. LGB (30%) + XGB (70%)  →  calcula probabilidad de fraude
        ↓
6. prob >= 0.3144  →  is_fraud = 1 (fraude detectado)
```

---

## Endpoints

### `GET /health`
**¿Para qué sirve?** Comprobar que la API está viva y funcionando.  
**¿Quién lo usa?** Full Stack puede llamarlo para saber si la API está caída.  
**No recibe ningún dato.**

```
Llamada:  GET http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "3.0.0",
  "modelo": "modelo_07_v1"
}
```

---

### `POST /predict`
**¿Para qué sirve?** Predecir si UNA transacción es fraude o no.  
**¿Quién lo usa?**
- **Ciber** → para atacar con transacciones sospechosas una a una
- **Full Stack** → cuando entra una transacción nueva desde su app

**Recibe:** JSON con los datos de la transacción.  
**Guarda automáticamente** el resultado en la base de datos.

```
Llamada:  POST http://localhost:8000/predict
Body:     JSON con los datos de la transacción
```

**Ejemplo respuesta transacción legítima:**
```json
{
  "id_transaccion"    : "tx-001",
  "is_fraud"          : 0,
  "prob_fraud"        : 0.2855,
  "impacto_fraude"    : 0,
  "es_transfronteriza": 0,
  "ratio_imp_limite"  : 0.075,
  "intensidad_tx"     : 0.0003,
  "severidad_tx"      : 150.0,
  "flujo_neto_30d"    : 200.0,
  "mensaje"           : "Transaccion legitima — probabilidad fraude 29%"
}
```

**Ejemplo respuesta transacción fraudulenta:**
```json
{
  "id_transaccion"    : "tx-002",
  "is_fraud"          : 1,
  "prob_fraud"        : 0.8734,
  "impacto_fraude"    : 3,
  "es_transfronteriza": 1,
  "ratio_imp_limite"  : 0.95,
  "intensidad_tx"     : 0.5,
  "severidad_tx"      : 4500.0,
  "flujo_neto_30d"    : -4500.0,
  "mensaje"           : "FRAUDE DETECTADO — probabilidad fraude 87%"
}
```

**Significado de los campos de respuesta:**

| Campo | Descripción |
|-------|-------------|
| `is_fraud` | 0 = legítima, 1 = fraude |
| `prob_fraud` | Probabilidad de fraude de 0.0 a 1.0 |
| `impacto_fraude` | 0=no fraude, 1=bajo(<500€), 2=medio(<2000€), 3=alto(>2000€) |
| `es_transfronteriza` | 0=misma país, 1=país diferente al del cliente |
| `ratio_imp_limite` | Importe / límite tarjeta (0.95 = usó el 95% del límite) |
| `intensidad_tx` | Num transacciones / tiempo — alto = sospechoso |
| `severidad_tx` | Importe × num transacciones — alto = sospechoso |
| `flujo_neto_30d` | Vol entrante − vol saliente — negativo = sale más de lo que entra |

---

### `POST /predict/batch`
**¿Para qué sirve?** Predecir MUCHAS transacciones a la vez.  
**¿Quién lo usa?**
- **Ciber** → para lanzar ataques masivos con muchas transacciones de golpe

**Recibe:** lista de transacciones en JSON entre corchetes `[ ]`.  
**Guarda automáticamente** todos los resultados en la Base de datos.

```
Llamada:  POST http://localhost:8000/predict/batch
Body:     [ { transaccion1 }, { transaccion2 }, { transaccion3 } ]
```

**Ejemplo respuesta con 3 transacciones:**
```json
{
  "total"   : 3,
  "fraudes" : 1,
  "legitimas": 2,
  "predicciones": [
    {
      "id_transaccion"    : "tx-001",
      "is_fraud"          : 0,
      "prob_fraud"        : 0.2855,
      "impacto_fraude"    : 0,
      "es_transfronteriza": 0,
      "ratio_imp_limite"  : 0.075,
      "intensidad_tx"     : 0.0003,
      "severidad_tx"      : 150.0,
      "flujo_neto_30d"    : 200.0,
      "mensaje"           : "Transaccion legitima — probabilidad fraude 29%"
    },
    {
      "id_transaccion"    : "tx-002",
      "is_fraud"          : 1,
      "prob_fraud"        : 0.8734,
      "impacto_fraude"    : 3,
      "es_transfronteriza": 1,
      "ratio_imp_limite"  : 0.95,
      "intensidad_tx"     : 0.5,
      "severidad_tx"      : 4500.0,
      "flujo_neto_30d"    : -4500.0,
      "mensaje"           : "FRAUDE DETECTADO — probabilidad fraude 87%"
    },
    {
      "id_transaccion"    : "tx-003",
      "is_fraud"          : 0,
      "prob_fraud"        : 0.1243,
      "impacto_fraude"    : 0,
      "es_transfronteriza": 0,
      "ratio_imp_limite"  : 0.04,
      "intensidad_tx"     : 0.0001,
      "severidad_tx"      : 80.0,
      "flujo_neto_30d"    : 500.0,
      "mensaje"           : "Transaccion legitima — probabilidad fraude 12%"
    }
  ]
}
```

---

### `GET /metrics  -- ELIMINADO`
**¿Para qué sirve?** Ver métricas del modelo en tiempo real.  
**¿Quién lo usa?**
- **Full Stack** → para mostrar el dashboard de KPIs al analista
- **Data Science** → para ver cómo está funcionando el modelo

**No recibe ningún dato.** Lee directamente de la base de datos.


```json
{
  "total_predicciones" : 150,
  "fraudes_detectados" : 23,
  "legitimas"          : 127,
  "tasa_deteccion"     : 0.1533,
  "prob_fraud_media"   : 0.3124,
  "pendientes_revision": 18,
  "confirmados_fraude" : 4,
  "falsos_positivos"   : 1
}
```

```
SELECT COUNT(*) AS total_predicciones,
  SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END)  AS fraudes_detectados,
  SUM(CASE WHEN is_fraud = 0 THEN 1 ELSE 0 END)  AS legitimas,
  ROUND(AVG(prob_fraud)::numeric, 4)             AS prob_fraud_media,
  SUM(CASE WHEN estado_revision = 'pendiente'THEN 1 ELSE 0 END) AS pendientes_revision,
  SUM(CASE WHEN estado_revision = 'confirmado_fraude' THEN 1 ELSE 0 END) AS confirmados_fraude,
  SUM(CASE WHEN estado_revision = 'falso_positivo' THEN 1 ELSE 0 END) AS falsos_positivos
  FROM transacciones
  WHERE is_fraud IS NOT NULL
```
---

## Campos que recibe la API

Los datos se envían con los **mismos nombres que están en la BD**:

| Grupo | Campo | Tipo | Ejemplo |
|-------|-------|------|---------|
| **Identificación** | id_transaccion | str | "059638c5-40f" |
| | id_cliente | str | "3ddebd45-ccd" |
| **Cliente** | tipo_cliente | str | "persona" |
| | edad_cliente | int | 35 |
| | customer_country | str | "ES" |
| | customer_region | str | "Centro" |
| | tenure | int | 365 |
| | importe_medio_mensual | float | 500.00 |
| | desviacion_estandar_mensual | float | 150.00 |
| | media_transacciones_al_dia | float | 3.5 |
| | numero_fraudes_ultimo_ano | int | 0 |
| **Cuenta** | id_cuenta | str | "8f3262c1-69a" |
| | cuenta_origen | str | "ES20427866183" |
| | estado_cuenta | str | "activa" |
| | saldo_actual | float | 2500.00 |
| | saldo_medio_30_dias | float | 2200.00 |
| | volumen_entrante_30_dias | float | 3000.00 |
| | volumen_saliente_30_dias | float | 2800.00 |
| | numero_transferencias_recibidas_7_dias | int | 3 |
| | numero_transferencias_enviadas_7_dias | int | 2 |
| **Tarjeta** | id_tarjeta | str | "00df680c-e19" |
| | estado_tarjeta | str | "activa" |
| | fecha_creacion_tarjeta | str | "2023-01-15" |
| | antiguedad_tarjeta_dias | int | 365 |
| | limite_importe_transacciones | float | 2000.00 |
| | veces_superar_limite_7_dias | int | 0 |
| **Transacción** | tipo_transaccion | str | "tarjeta" |
| | fecha_hora | str | "2026-05-23T14:30:00" |
| | is_night | int | 0 |
| | is_weekend | int | 0 |
| | tiempo_desde_ultima_transaccion | int | 3600 |
| | numero_transacciones_ultima_hora | int | 1 |
| | importe_transaccion | float | 150.00 |
| | metodo_autenticacion | str | "PIN" |
| | numero_pin_disponibles | int | 3 |
| | identificador_dispositivo_fingerprint | str | "fa1bdf50" |
| | dispositivo_reconocido | int | 1 |
| | operacion_pais | str | "ES" |
| | operacion_region | str | "Centro" |
| | direccion_ip_origen | str | "86.34.12.179" |
| | geolocalizacion | str | "40.4168,-3.7038" |
| | cuenta_destino | str | "ES169540317577" |
| | destino_alto_riesgo | int | 0 |

---


