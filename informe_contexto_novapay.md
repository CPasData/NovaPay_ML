# NovaPay ML — Informe de Contexto del Proyecto
**Desafío de Tripulaciones · Operación Centinela**  
**Equipo Data Science — Blue Team, Grupo 2**  
**Versión API: 3.0.0 · Mayo 2026**

---

## 1. ¿Qué hemos construido?

Un sistema de **detección de fraude bancario en tiempo real** para NovaPay. Cuando entra una transacción, nuestra API la analiza con un modelo de Machine Learning (LightGBM + XGBoost) y devuelve en milisegundos si es fraudulenta, con qué probabilidad y cuál es el impacto económico estimado.

El sistema está diseñado en dos rondas:

- **Ronda 1** — El modelo detecta fraudes con lo que ha aprendido durante el entrenamiento. Los analistas revisan y confirman/descartan cada caso.
- **Ronda 2** — Con las etiquetas reales que dejaron los analistas, el modelo se reentrena automáticamente y mejora su precisión antes de que Ciber vuelva a atacar.

---

## 2. Arquitectura del sistema

```
TRANSACCIÓN ENTRA AL SISTEMA
         ↓
┌────────────────────────────────────────┐
│  FULL STACK                            │
│  POST /predict                         │
│  (envía datos de la transacción)       │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  DATA SCIENCE — API FastAPI            │
│  1. Aplica el modelo ML                │
│  2. Calcula is_fraud + prob_fraud      │
│  3. Guarda TODO en PostgreSQL          │
│     → is_fraud, prob_fraud             │
│     → impacto_fraude                   │
│     → campos calculados del modelo     │
│     → estado_revision = 'pendiente'    │
│  4. Devuelve resultado a Full Stack    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  FULL STACK                            │
│  Muestra transacciones sospechosas     │
│  ordenadas por prob_fraud              │
│  El analista revisa cada caso          │
└────────────────────────────────────────┘
         ↓
   ┌─────┴─────┐
   ↓           ↓
FRAUDE      NO ES FRAUDE
"Confirmar" "Falso positivo"
   ↓           ↓
   └─────┬─────┘
         ↓
┌────────────────────────────────────────┐
│  FULL STACK actualiza en BD:           │
│  → target_final = TRUE/FALSE           │
│  → estado_revision                     │
│  → id_usuario, fecha_revision          │
└────────────────────────────────────────┘
```

---

## 3. Flujo de reentrenamiento (Ronda 2)

Cuando el analista termina de revisar todos los casos pendientes, Full Stack activa el botón **"Reentrenar modelo"** (solo visible cuando `pendientes_revision = 0` en `/metrics`):

```
Full Stack llama a:  POST /retrain
         ↓
Data Science (automático):
  1. Lee transacciones con target_final IS NOT NULL de la BD
  2. Reentrena con etiquetas REALES del analista
  3. Genera modelo_ronda2.pkl
  4. Reemplaza el .pkl activo en la API
  5. Reinicia con el modelo mejorado
         ↓
API responde:
{
  "estado"           : "modelo actualizado ✅",
  "precision_ronda1" : "65%",
  "precision_ronda2" : "82%"
}
```

> **Nota:** El endpoint `POST /retrain` está pendiente de implementar. Se implementará al final de Ronda 1 cuando tengamos etiquetas reales confirmadas por el analista.

---

## 4. Responsabilidades por equipo

| Tarea | Equipo |
|-------|--------|
| Llamar a `POST /predict` (o `/predict/batch`) | Full Stack |
| Predecir con el modelo ML | Data Science |
| Guardar predicción en BD automáticamente | Data Science |
| Mostrar casos sospechosos al analista | Full Stack |
| Actualizar `target_final` y `estado_revision` (endpoint `PUT /review`) | Full Stack |
| Mostrar botón "Reentrenar" cuando `pendientes = 0` | Full Stack |
| Llamar a `POST /retrain` cuando el analista termina | Full Stack |
| Reentrenar y activar el modelo mejorado | Data Science |
| Atacar la API con transacciones sospechosas | Ciberseguridad |

---

## 5. La API — Endpoints

La API está corriendo en FastAPI con documentación interactiva disponible en `http://localhost:8000/docs`.

### `GET /health`
Comprueba que la API está activa y el modelo cargado.

```
Llamada: GET http://localhost:8000/health
```
```json
{
  "status"   : "ok",
  "version"  : "3.0.0",
  "modelo"   : "modelo_07_v1",
  "ensemble" : "LGB 30% + XGB 70%",
  "threshold": 0.3144
}
```

---

### `POST /predict`
Recibe **una** transacción, devuelve la predicción y guarda en BD.

**¿Quién lo usa?**
- **Ciber** → para atacar transacción a transacción
- **Full Stack** → cuando entra una transacción desde la app

```
Llamada: POST http://localhost:8000/predict
Body:    JSON con los datos de la transacción
```

**Respuesta — transacción legítima:**
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

**Respuesta — fraude detectado:**
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
| `prob_fraud` | Probabilidad de fraude (0.0 a 1.0) |
| `impacto_fraude` | 0=no fraude · 1=bajo (<500€) · 2=medio (<2000€) · 3=alto (≥2000€) |
| `es_transfronteriza` | 1 si la operación es desde un país diferente al del cliente |
| `ratio_imp_limite` | Importe / límite tarjeta (0.95 = usó el 95% del límite) |
| `intensidad_tx` | Número de transacciones / tiempo — alto es sospechoso |
| `severidad_tx` | Importe × número de transacciones — alto es sospechoso |
| `flujo_neto_30d` | Vol. entrante − vol. saliente — negativo = sale más de lo que entra |

---

### `POST /predict/batch`
Recibe **muchas** transacciones a la vez y devuelve predicciones para todas. Guarda cada resultado en BD.

**¿Quién lo usa?**
- **Ciber** → para lanzar ataques masivos con muchas transacciones de golpe

```
Llamada: POST http://localhost:8000/predict/batch
Body:    [ { transaccion1 }, { transaccion2 }, ... ]
```

```json
{
  "total"       : 3,
  "fraudes"     : 1,
  "legitimas"   : 2,
  "predicciones": [ ... ]
}
```

---

### `GET /metrics`
Devuelve métricas del modelo en tiempo real, calculadas directamente desde la BD. No recibe datos.

**¿Quién lo usa?**
- **Full Stack** → para el dashboard de KPIs del analista (incluida la señal de cuántos `pendientes_revision` quedan)

```
Llamada: GET http://localhost:8000/metrics
```
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

> El campo clave para Full Stack es `pendientes_revision`: cuando llega a **0**, se activa el botón "Reentrenar modelo".

---

## 6. Datos que recibe la API (campos de entrada)

Los campos se envían con los **mismos nombres que están en la BD**:

| Grupo | Campo | Tipo | Ejemplo |
|-------|-------|------|---------|
| **Identificación** | `id_transaccion` | str | `"059638c5-40f"` |
| | `id_cliente` | str | `"3ddebd45-ccd"` |
| **Cliente** | `tipo_cliente` | str | `"persona"` |
| | `edad_cliente` | int | `35` |
| | `customer_country` | str | `"ES"` |
| | `customer_region` | str | `"Centro"` |
| | `tenure` | int | `365` |
| | `importe_medio_mensual` | float | `500.00` |
| | `desviacion_estandar_mensual` | float | `150.00` |
| | `media_transacciones_al_dia` | float | `3.5` |
| | `numero_fraudes_ultimo_ano` | int | `0` |
| **Cuenta** | `id_cuenta` | str | `"8f3262c1-69a"` |
| | `cuenta_origen` | str | `"ES20427866183"` |
| | `estado_cuenta` | str | `"activa"` |
| | `saldo_actual` | float | `2500.00` |
| | `saldo_medio_30_dias` | float | `2200.00` |
| | `volumen_entrante_30_dias` | float | `3000.00` |
| | `volumen_saliente_30_dias` | float | `2800.00` |
| | `numero_transferencias_recibidas_7_dias` | int | `3` |
| | `numero_transferencias_enviadas_7_dias` | int | `2` |
| **Tarjeta** | `id_tarjeta` | str | `"00df680c-e19"` |
| | `estado_tarjeta` | str | `"activa"` |
| | `fecha_creacion_tarjeta` | str | `"2023-01-15"` |
| | `antiguedad_tarjeta_dias` | int | `365` |
| | `limite_importe_transacciones` | float | `2000.00` |
| | `veces_superar_limite_7_dias` | int | `0` |
| **Transacción** | `tipo_transaccion` | str | `"tarjeta"` |
| | `fecha_hora` | str | `"2026-05-23 14:30:00"` |
| | `is_night` | int | `0` |
| | `is_weekend` | int | `0` |
| | `tiempo_desde_ultima_transaccion` | int | `3600` |
| | `numero_transacciones_ultima_hora` | int | `1` |
| | `importe_transaccion` | float | `150.00` |
| | `metodo_autenticacion` | str | `"PIN"` |
| | `numero_pin_disponibles` | int | `3` |
| | `identificador_dispositivo_fingerprint` | str (opcional) | `"fa1bdf50"` |
| | `dispositivo_reconocido` | int | `1` |
| | `operacion_pais` | str | `"ES"` |
| | `operacion_region` | str | `"Centro"` |
| | `direccion_ip_origen` | str (opcional) | `"86.34.12.179"` |
| | `geolocalizacion` | str (opcional) | `"40.4168,-3.7038"` |
| | `cuenta_destino` | str (opcional) | `"ES169540317577"` |
| | `destino_alto_riesgo` | int | `0` |

---

## 7. Lo que guarda la API en Base de Datos

Cada llamada a `/predict` o `/predict/batch` guarda automáticamente en la tabla `transacciones` todos los campos de entrada más los siguientes campos calculados:

**Campos que calcula el modelo (Data Science):**

| Campo | Descripción |
|-------|-------------|
| `is_fraud` | 0=legítima, 1=fraude |
| `prob_fraud` | Probabilidad de fraude |
| `impacto_fraude` | 0=no fraude · 1=bajo · 2=medio · 3=alto |
| `es_transfronteriza` | Operación desde otro país |
| `ratio_imp_limite` | Importe / límite tarjeta |
| `intensidad_tx` | Intensidad de transacciones |
| `severidad_tx` | Importe × número de transacciones |
| `flujo_neto_30d` | Vol. entrante − vol. saliente |
| `estado_revision` | `'pendiente'` si fraude · `'legitima'` si no |

**Campos que rellena el analista vía Full Stack:**

| Campo | Descripción |
|-------|-------------|
| `target_final` | `TRUE` = fraude confirmado · `FALSE` = falso positivo |
| `estado_revision` | `'confirmado_fraude'` / `'falso_positivo'` / `'legitima'` |
| `id_usuario` | Analista que revisó |
| `fecha_revision` | Cuándo fue revisada |

> Si hay conflicto de `id_transaccion` (transacción ya existente), la API ignora el duplicado con `ON CONFLICT DO NOTHING`.

---

## 8. El modelo ML

El modelo está guardado en un único archivo `.pkl` que contiene todo lo necesario para hacer predicciones:

```
modelo_07_v1.pkl
├── fe         → FeatureEngineer v3 (genera 67 features automáticamente)
├── imputer    → KNNImputer n_neighbors=5 (rellena valores nulos)
├── scaler     → StandardScaler (escala los datos)
├── lgb_model  → LightGBM
├── xgb_model  → XGBoost
├── best_w     → peso del ensemble (LGB ~30%, XGB ~70%)
├── best_t     → threshold de decisión (~0.3144)
└── num_feats  → lista de features numéricas para imputer y scaler
```

**Pipeline de predicción en la API (orden exacto):**

```
Transacción entra a la API
        ↓
1. FeatureEngineer.transform()
   → genera 67 features a partir de los 37 campos de entrada
        ↓
2. Extrae campos calculados antes de escalar
   → es_transfronteriza, ratio_imp_limite, intensidad_tx,
     severidad_tx, flujo_neto_30d
        ↓
3. KNNImputer
   → rellena valores nulos en las 67 features numéricas
        ↓
4. StandardScaler
   → escala los números
        ↓
5. LGB × best_w  +  XGB × (1−best_w)
   → calcula probabilidad ensemble de fraude
        ↓
6. prob_fraud >= best_t  →  is_fraud = 1 (FRAUDE DETECTADO)
```

**Features que genera el FeatureEngineer (67 en total):**
- Ratios e indicadores financieros (límite, saldo, flujos)
- Flags geográficos (operación transfronteriza, región)
- Transformaciones logarítmicas (importes, saldos)
- Encoding de frecuencia + target encoding para variables categóricas (16 features)
- Features de sesión/burst (5): `txn_por_minuto`, `burst_rapido`, `alta_velocidad`, `monto_velocidad`, `tiempo_ultima_bin`
- Desviaciones temporales por cliente (3): `diff_hora_cliente`, `diff_importe_cliente`, `ratio_actividad_cliente`
- Red de destinos (1): `frecuencia_destino`
- Flags compuestos (4): `foreign_unknown_device`, `night_velocity`, `high_ratio_redondeado`, `same_country_device`

**Rendimiento del modelo (evaluado en test set 20%):**

| Modelo | Dataset | PR-AUC | AUC-ROC | Precisión | Recall |
|--------|---------|--------|---------|-----------|--------|
| `modelo_07_v1.pkl` | v1 (señal débil) | 0.3236 | 0.7135 | 19.7% | 87.2% |
| `modelo_08_v2.pkl` | v2 (señal fuerte) | **0.9640** | **0.9874** | **77.2%** | **95.3%** |

> La métrica principal es **PR-AUC** (área bajo la curva Precisión-Recall), más adecuada que AUC-ROC para datasets con fraude desbalanceado.

---

## 9. Cómo arrancar la API en local

**Requisitos previos:**
- Docker corriendo con el contenedor `postgres-demo`
- BD `novapay` creada con la tabla `transacciones`

```bash
# Desde la raíz del proyecto
python app.py
```

La API estará disponible en:
- **URL base:** `http://localhost:8000`
- **Swagger UI (documentación interactiva):** `http://localhost:8000/docs`

**Verificar que funciona:**
```bash
curl http://localhost:8000/health
```

---

## 10. Conexión a Supabase

Cuando Full Stack tenga Supabase lista, Data Science solo cambia el bloque `DB_CONFIG` en `app.py`:

```python
# Ahora (local):
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 5432,
    "database": "novapay",
    "user"    : "postgres",
    "password": "123456"
}

# Cuando Supabase esté lista:
DB_CONFIG = {
    "host"    : "db.xxxxxxxxxxxx.supabase.co",
    "port"    : 5432,
    "database": "postgres",
    "user"    : "postgres",
    "password": "contraseña_supabase"
}
```

El resto del código no cambia.

---

## 11. Archivos principales del proyecto

| Archivo | Descripción |
|---------|-------------|
| `app.py` | API FastAPI — corazón del proyecto. Carga el modelo, define los 4 endpoints, pipeline de predicción y guardado en BD |
| `scripts/feature_engineering.py` | `FeatureEngineer` v3 — genera las 67 features a partir de los datos brutos |
| `model/modelo_07_v1.pkl` | Modelo entrenado con v1 (actualmente en uso) |
| `model/modelo_08_v2.pkl` | Modelo entrenado con v2 — PR-AUC 0.96 (pendiente de activar) |
| `Notebooks/09_pipeline_completo.ipynb` | Notebook principal — pipeline completo, parametrizable con `DATASET='v1'` o `'v2'` |
| `test_api.ipynb` | Notebook de pruebas para los 4 endpoints |
| `dockerfile` | Build multi-stage con Python 3.11-slim y Uvicorn |
| `docker-compose.yml` | Puerto 8000, hot-reload en `app.py`, monta `./model` |
| `docs/contexto_ml.md` | Documentación técnica extendida del proyecto ML |

---

## 12. Pendientes

| Tarea | Equipo | Estado |
|-------|--------|--------|
| Activar `modelo_08_v2.pkl` en lugar de v1 (PR-AUC 0.96 vs 0.32) | Data Science | Pendiente |
| Despliegue en AWS EC2 con Docker | Data Science | Pendiente |
| Crear BD en Supabase | Full Stack | Pendiente |
| Conectar `app.py` a Supabase (cambiar `DB_CONFIG`) | Data Science | Pendiente URL de Supabase |
| Endpoint `PUT /review` — actualizar revisión del analista | Full Stack (Node) | Pendiente confirmación |
| Endpoint `POST /retrain` — reentrenamiento Ronda 2 | Data Science | Pendiente a Ronda 1 |

---

*Última actualización: Mayo 2026 — Blue Team, Grupo 2*
