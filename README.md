# NovaPay ML — Operación Centinela

<div align="center">
  <p>Sistema de detección de fraude transaccional en tiempo real</p>
  <p>Ensemble LightGBM + XGBoost · 69 features · 3% fraude · recall@k · thresholds por canal</p>
</div>

---

## Para negocio / stakeholders

### ¿Qué problema resuelve?

NovaPay procesa ~200.000 transacciones al día. Aproximadamente un 3% (~6.000) son fraudulentas, pero el equipo de analistas solo puede revisar ~1.000 alertas al día. Sin un sistema automático, la mayoría de fraudes pasan desapercibidos o requieren un equipo humano inasumible.

**El problema tiene 3 dimensiones:**

| Dimensión | Impacto |
|---|---|
| **Volumen** | 200K tx/día es imposible de revisar manualmente |
| **Velocidad** | Cada transacción debe evaluarse en <100ms para no bloquear la operativa |
| **Precisión** | Alertas falsas saturan a los analistas; fraudes no detectados son pérdida directa |

### ¿Qué valor aporta?

| Indicador | Valor v3 | Qué significa |
|---|---|---|
| **Fraudes bloqueados automáticamente** | ~71.8% | El modelo bloquea sin intervención humana (~4.300 fraudes/día) |
| **Fraudes recuperados vía analistas** | +12.9% | Analistas revisan el top k y recuperan ~774 fraudes más |
| **Recall total** | 84.7% | De cada 100 fraudes, 85 son detectados (bloqueo + revisión) |
| **Falsos positivos** | ~21.3% | 1 de cada 5 alertas requiere liberación manual (~30s cada una) |
| **Tiempo de inferencia** | ~50ms | La decisión llega antes de que el cliente termine la operación |
| **ROI estimado** | 3.100:1 | Por cada euro invertido en analistas se recuperan ~3.100€ en fraude |

### ¿Cómo se usa en la práctica?

**Flujo operativo:**

```
Transacción → API /predict → Feature Engineering → Ensemble → Decisión
                                                                │
                    ┌───────────────────────────────────────────┤
                    ▼                                           ▼
            is_fraud = 1 (prob ≥ threshold)           is_fraud = 0
                    │                                           │
        ┌───────────┴───────────┐                       Pasa sin revisión
        ▼                       ▼
  Bloqueo automático     Alerta para analista
  (top fraudes,          (revisa cola priorizada
   alta prob)             por prob_fraud)
```

**La API devuelve 10 campos por transacción:**

| Campo | Tipo | Qué indica |
|---|---|---|
| `is_fraud` | 0/1 | Decisión del modelo |
| `prob_fraud` | 0.0–1.0 | Probabilidad de fraude |
| `impacto_fraude` | 0–3 | Bajo/Medio/Alto según importe |
| `es_transfronteriza` | 0/1 | ¿Operación en país distinto al del cliente? |
| `ratio_imp_limite` | float | % del límite de tarjeta usado |
| `intensidad_tx` | float | Transacciones por minuto (detección de ráfagas) |
| `severidad_tx` | float | importe × número de transacciones |
| `flujo_neto_30d` | float | Diferencia entre ingresos y gastos mensuales |
| `mensaje` | str | "FRAUDE DETECTADO" o "Transacción legítima" |
| `id_transaccion` | str | Identificador único |

### ¿Qué métricas importan?

| Métrica | v3 | Target | Notas |
|---|---|---|---|
| **PR-AUC** | 0.90 | >0.85 | Principal métrica de ranking (baseline 3.5%) |
| **AUC-ROC** | 0.987 | >0.97 | Separación fraude/legítimo |
| **F2-score** | 0.67 | >0.60 | Prioriza recall sobre precisión (decisión de negocio) |
| **Recall global** | 84.7% | >80% | % de fraudes detectados (bloqueados + revisados) |
| **Recall@k (k=0.5%)** | 12.9% | >10% | % de fraudes capturados entre las 1.000 alertas revisables |
| **Precisión** | 78.7% | >75% | % de alertas que son fraude real |
| **Brier Score** | 0.033 | <0.05 | Calibración de probabilidades |
| **ECE** | 0.062 | <0.05 | Error de calibración (mejorable con recalibración isotónica) |

---

## Para tech / desarrolladores

### Stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| **Lenguaje** | Python | 3.11+ |
| **API** | FastAPI + Uvicorn | 0.136 + 0.047 |
| **Validación** | Pydantic v2 | 2.13 |
| **ML Core** | scikit-learn | 1.8 |
| **Gradient Boosting** | LightGBM, XGBoost | — |
| **Serialización** | joblib | 1.5 |
| **Datos** | pandas, numpy, scipy | 3.0, 2.4, 1.17 |
| **Container** | Docker + docker-compose | — |
| **BD (futuro)** | PostgreSQL | — |

### Estructura del proyecto

```
C:\Dev\NovaPay_ML\
│
├── app.py                      # FastAPI — endpoints /predict, /health
├── dockerfile                  # Imagen Docker para producción
├── docker-compose.yml          # Orquestación local
├── requirements.txt            # Dependencias Python
├── requirements-doc.txt        # Dependencias para doc (sin lightgbm)
│
├── scripts/                    # Código principal
│   ├── feature_engineering.py  # FeatureEngineer (69 features, v4)
│   ├── regenerate_models.py    # Entrenamiento completo v1+v2+v3
│   ├── prediccion_lote.py      # Inferencia batch CSV → CSV
│   ├── inference_example.py    # Ejemplo de carga + inferencia
│   ├── evaluacion_rondas.py    # Simulación de producción con drift
│   ├── generar_muestra_sin_etiqueta.py  # Datos sin target (3 perfiles)
│   ├── generar_lote_prediccion.py       # Combina datasets para inferencia
│   ├── separar_prediccion_json.py       # CSV → JSON separado
│   ├── aplanar_predicciones.py          # Aplana a 10 campos
│   │
│   └── synthetic_data/         # Generadores de datos sintéticos
│       ├── generar_dataset_fraude.py      # v1 (señal débil)
│       ├── generar_dataset_fraude_v2.py   # v2 (señal mejorada)
│       └── generar_dataset_fraude_v3.py   # v3 (3% fraude, bizum, 100K)
│
├── model/                      # Modelos entrenados (.pkl autocontenidos)
│   ├── modelo_07_v1.pkl        # v1 — 15% fraude, señal débil
│   ├── modelo_08_v2.pkl        # v2 — 15% fraude, señal mejorada
│   └── modelo_09_v3.pkl        # v3 — 3% fraude, thresholds por canal
│
├── data/                       # Datasets
│   ├── dataset_fraude.csv      # v1 (10K tx)
│   ├── dataset_fraude_v2.csv   # v2 (10K tx)
│   ├── dataset_fraude_v3.csv   # v3 (100K tx, dataset principal)
│   └── muestra_*.csv/.json     # Muestras sin etiqueta para pruebas
│
├── notebooks/                  # Jupyter notebooks (entrenamiento + EDA)
│   └── 09_pipeline_completo.ipynb  # Pipeline completo v3
│
└── docs/                       # Documentación
    ├── guion_presentacion.md   # Guión completo de presentación
    ├── comandos.md             # Todos los comandos del proyecto
    ├── contexto_ml.md          # Contexto técnico detallado
    ├── preguntas_posibles_presentacion.md  # FAQ para presentación
    └── esquema_bd.sql          # Esquema de base de datos
```

### Cómo instalar y ejecutar

#### 1. Requisitos

- Python 3.11 o superior
- Docker (opcional, para producción)
- Git

#### 2. Clonar e instalar

```powershell
# Entorno virtual con uv
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

#### 3. Generar datos (opcional, ya incluidos)

```powershell
python scripts/synthetic_data/generar_dataset_fraude_v3.py
```

#### 4. Entrenar modelos (opcional, ya incluidos)

```powershell
python scripts/regenerate_models.py
```

#### 5. Arrancar la API

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Probar

```powershell
# Health check
curl http://localhost:8000/health

# Predecir una transacción
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d (Get-Content data/muestra_sin_etiqueta.json -Raw | ConvertTo-Json -Compress)

# Swagger UI: http://localhost:8000/docs
```

#### 7. Con Docker

```powershell
# Construir imagen
docker build -t novapay-ml:v3 -f dockerfile .

# Ejecutar
docker run -d --name novapay-api -p 8000:8000 --restart unless-stopped novapay-ml:v3

# O con compose (incluye hot-reload)
docker compose up -d
```

#### 8. Todo el pipeline completo

```powershell
python scripts/synthetic_data/generar_dataset_fraude_v3.py   # 1. Generar datos
python scripts/regenerate_models.py                          # 2. Entrenar modelo
python scripts/generar_muestra_sin_etiqueta.py --perfil todo # 3. Muestras de prueba
python scripts/prediccion_lote.py --input data/muestra_mixto.csv --output data/predicciones.csv  # 4. Inferencia batch
uvicorn app:app --host 0.0.0.0 --port 8000                   # 5. API
python scripts/evaluacion_rondas.py --modelo v3 --drift suave # 6. Simular producción
```

### Cómo contribuir

**Workflow recomendado:**

1. **Explorar**: los notebooks en `notebooks/` contienen el historial completo de experimentación
2. **Generar datos**: modificar `generar_dataset_fraude_v3.py` para añadir nuevos patrones de fraude
3. **Feature engineering**: editar `scripts/feature_engineering.py` (clase `FeatureEngineer`)
4. **Entrenar**: ejecutar `regenerate_models.py` para regenerar los 3 modelos
5. **Validar**: revisar `docs/guion_presentacion.md` (sección 5 — Métricas) para umbrales aceptables
6. **Documentar**: actualizar `docs/` con cada cambio significativo

**Reglas:**

- No añadir dependencias sin actualizar `requirements.txt` y `requirements-doc.txt`
- Los `.pkl` deben ser autocontenidos (incluir `FeatureEngineer`, scaler, imputer, ensemble, thresholds)
- Preservar los 10 campos de salida de la API
- Todos los scripts deben ser ejecutables desde la raíz del proyecto
- No eliminar `scripts/feature_engineering.py` (el `.pkl` lo referencia para deserializar)

### Modelo: arquitectura interna

Cada `.pkl` contiene un pipeline completo:

```
modelo.pkl = {
    'fe':      FeatureEngineer      # sklearn Transformer → 44 → 69 features
    'scaler':  StandardScaler       # Normaliza features numéricas
    'imputer': KNNImputer (n=5)     # Imputa nulos por similitud
    'lgb_model': LightGBM           # Boosting rápido, captura outliers
    'xgb_model': XGBoost           # Boosting robusto, maneja desbalance
    'best_w':   float               # Peso del ensemble (ej: 0.65 = 65% LGB)
    'best_t':   float               # Threshold global (ej: 0.7622)
    'num_feats': list[str]          # Features numéricas para scaler/imputer
    'per_channel_thresholds': dict  # Threshold por canal (tarjeta/transferencia/bizum)
    'metadata': dict                # Métricas adicionales (recall@k, etc.)
}
```

**Pipeline de inferencia en producción (`app.py`):**

```
POST /predict
  → JSON → Pydantic Transaccion (validación: importe>0, flags 0/1, etc.)
  → pd.DataFrame
  → FeatureEngineer.transform()    (69 features)
  → Extraer campos para respuesta  (cross_border, intensidad, etc.)
  → StandardScaler.transform()      (solo numéricas)
  → KNNImputer.transform()         (solo numéricas)
  → LightGBM.predict_proba()        (probabilidad LGB)
  → XGBoost.predict_proba()        (probabilidad XGB)
  → Ensemble ponderado: w*prob_lgb + (1-w)*prob_xgb
  → Threshold por canal: prob ≥ threshold_canal[tipo_transaccion]
  → Respuesta: 10 campos
```

### Datos: las 69 features (v4)

Las features transformadas se organizan en 4 bloques:

| Bloque | Features | Ejemplo |
|---|---|---|
| **Ratios financieros** (8) | `txn_vs_limit_pct`, `outflow_inflow_ratio`, `net_flow_30d`, `balance_utilization`, `saldo_ratio_ingreso`, `saldo_ratio_limite`, `txn_ratio_media` | % del límite usado en una transacción |
| **Flags geográficos** (4) | `cross_border`, `foreign_unknown_device`, `foreign_known_device`, `domestic_unknown_device` | Cruce país + dispositivo → 43.6% de fraudes |
| **Features de sesión** (7) | `txn_por_minuto`, `burst_rapido`, `alta_velocidad`, `txn_intensity`, `txn_severity`, `actividad_alta`, `ratio_dispositivo_hora` | Ráfagas de >5 tx en <5 min |
| **Desviación temporal** (7) | `diff_importe_zscore`, `diff_importe_signed`, `importe_anomalo`, `dias_desde_ultimo_pin`, `dias_desde_ultimo_cambio`, `antiguedad_relativa_media`, `diff_antiguedad_tx` | z-score real con media y desviación del cliente |

Las 69 features restantes son codificaciones target, frecuencias, flags categóricos y combinaciones de sesión.

### Pruebas de concepto

Para ver el modelo en acción:

```powershell
# 1. Generar muestra con perfil fraudulento (muchas señales de fraude)
python scripts/generar_muestra_sin_etiqueta.py --perfil fraude --n 100

# 2. Inferencia batch
python scripts/prediccion_lote.py --input data/muestra_fraude.csv --output data/predicciones_fraude.csv

# 3. O lanzar API y probar con curl
uvicorn app:app --host 0.0.0.0 --port 8000
# (en otra terminal)
curl -X POST http://localhost:8000/predict/batch `
  -H "Content-Type: application/json" `
  -d (Get-Content data/muestra_fraude.json -Raw)
```

### Versiones

| Versión | Fecha | Cambios |
|---|---|---|
| v1 | — | Señal débil, 15% fraude, threshold global, 67 features |
| v2 | — | Inyección de señal post-hoc, PR-AUC 0.35 → 0.96 |
| v3 | Actual | 3% fraude, 100K registros, 69 features, thresholds por canal, recall@k, z-score real, Pydantic validators |
| v3.1 | Roadmap | Recalibración isotónica (ECE <0.05), SHAP analysis |
| v4 | Futuro | Hard negative mining, datos reales, A/B testing |
