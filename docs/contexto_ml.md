# Proyecto NovaPay ML — Documento de Contexto Completo

Este documento describe la totalidad del proyecto de detección de fraude en transacciones
financieras de NovaPay. Está escrito para que otro asistente (LLM) pueda retomar el trabajo
sin necesidad de preguntar al usuario.

---

## 1. Propósito del Proyecto

Construir un pipeline de machine learning para detectar **fraude en transacciones bancarias**
usando datos sintéticos generados con Faker. El proyecto incluye:

- Generación de datos sintéticos transaccionales multi-nivel (clientes, cuentas, tarjetas, transacciones)
- Feature engineering con codificación target, frecuencias, detección de ráfagas y desviaciones de perfil
- Pipeline completo: FE v3 → KNNImputer → StandardScaler → LightGBM + XGBoost ensemble → threshold F2
- Modelos guardados (`.pkl`) conteniendo pipeline completo para inferencia
- Documentación de resultados

**Entorno**: Windows, Python 3.14, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0

---

## 2. Estructura del Proyecto

```
C:\Dev\NovaPay_ML\
├── scripts/                         # Código principal
│   ├── feature_engineering.py       # FE v3 (67 features, session/red/temporales)
│   ├── regenerate_models.py         # Regenera ambos .pkl + copia .py a saved_models/
│   ├── inference_example.py         # Ejemplo de inferencia con modelo guardado
│   ├── prediccion_lote.py           # Batch inference: input CSV → output CSV + predicciones
│   ├── evaluacion_rondas.py         # Evaluación por rondas de 100 txns (simula producción, detecta drift)
│   ├── generar_muestra_sin_etiqueta.py  # Genera CSV+JSON sin IS_FRAUD con perfiles (mixto/sospechoso/fraude)
│   ├── save_models.py               # Legacy (reemplazado por regenerate_models.py)
│   ├── __init__.py
│   ├── saved_models/
│   │   ├── feature_engineering.py   # Copia autónoma para carga sin scripts/ en sys.path
│   │   ├── modelo_07_v1.pkl         # Pipeline FE v3 + Scaler + Imputer + LGB + XGB + w + thr
│   │   └── modelo_08_v2.pkl         # Idem, entrenado con v2
│   └── synthetic_data/
│       ├── generar_dataset_fraude.py          # Generador v1 (señal débil)
│       ├── generar_dataset_fraude_v2.py # Generador v2 (señal fuerte)
│       └── sdv/
│           └── generar_dataset_sdv.py         # Generador alternativo con SDV
│
├── notebooks/
│   ├── 07_train_fraud_v1.ipynb      # Entrenamiento v1 (LGB + XGB)
│   ├── 08_train_fraud_v2.ipynb      # Entrenamiento v2 (LGB + XGB)
│   ├── 09_pipeline_completo.ipynb   # Pipeline completo FE v3 + Ensemble + F2 thr
│   ├── 09_pipeline_completo_v1.ipynb # Pipeline completo sobre v1
│   ├── EDA.ipynb                    # Análisis exploratorio inicial
│   ├── EDA_v2.ipynb                 # Análisis exploratorio ronda 2
│   └── docs/
│       └── contexto_eda.md          # Documentación del EDA
│
├── data/
│   ├── dataset_fraude.csv           # v1 (10K txns, ~15% fraude)
│   └── dataset_fraude_v2.csv  # v2 (10K txns, ~15% fraude, señal mejorada)
│
├── docs/
│   ├── contexto_ml.md               # ← ESTE DOCUMENTO
│   ├── mejora_senal_fraude.md       # Documentación de mejora de señal
│   ├── residuos_y_evaluacion.md     # Análisis de residuos y técnicas de evaluación
│   ├── esquema_bd.sql               # Esquema de base de datos
│   ├── 03_feature_engineering_deep_dive.ipynb  # Análisis v3 (sesión/red/temporales)
│   └── 06_model_selection_deep_dive.ipynb      # Comparación de modelos + ensemble
│
├── app.py                           # API FastAPI para inferencia
└── README.md
```

---

## 3. Datos Sintéticos

### 3.1 Dataset generado

Genera **10,000 transacciones** con estructura jerárquica:

| Nivel | Entidad | Cantidad | Columnas clave |
|-------|---------|----------|----------------|
| 1 | Cliente | ~2,000 | `tipo_cliente`, `edad_cliente`, `customer_country`, `tenure`, `importe_medio_mensual` |
| 2 | Cuenta | ~4,500 | `estado_cuenta`, `saldo_actual`, `volumen_entrante/saliente` |
| 3 | Tarjeta | ~9,000 | `estado_tarjeta`, `antiguedad_tarjeta_dias`, `limite_importe_transacciones` |
| 4 | Transacción | 10,000 | `importe_transaccion`, `is_night`, `dispositivo_reconocido`, `destino_alto_riesgo` |

**Columnas del dataset** (44 columnas, iguales en v1 y v2):

```
id_cliente, tipo_cliente, edad_cliente,
customer_country, customer_region, tenure,
importe_medio_mensual, desviacion_estandar_mensual,
media_transacciones_al_dia, numero_fraudes_ultimo_ano,
id_cuenta, cuenta_origen, estado_cuenta,
saldo_actual, saldo_medio_30_dias,
volumen_entrante_30_dias, volumen_saliente_30_dias,
numero_transferencias_recibidas_7_dias, numero_transferencias_enviadas_7_dias,
id_tarjeta, estado_tarjeta, fecha_creacion_tarjeta,
antiguedad_tarjeta_dias, limite_importe_transacciones,
veces_superar_limite_7_dias,
id_transaccion, tipo_transaccion, fecha_hora,
is_night, is_weekend,
tiempo_desde_ultima_transaccion, numero_transacciones_ultima_hora,
importe_transaccion, metodo_autenticacion, numero_pin_disponibles,
identificador_dispositivo_fingerprint, dispositivo_reconocido,
operacion_pais, operacion_region,
direccion_ip_origen, geolocalizacion,
cuenta_destino, destino_alto_riesgo,
IS_FRAUD, IMPACTO_FRAUDE
```

### 3.2 Generador v1 (`generar_dataset_fraude.py`)

- **Señal débil**: PR-AUC ~0.32, AUC-ROC ~0.71
- **Tasa de fraude**: ~15%
- **Mecanismo**: probabilidad aditiva con 15+ factores, cada uno contribuye 2-25%. Las features se generan independientemente con la misma distribución para todas las transacciones.
- **Problema**: la única dependencia features → label es la fórmula de probabilidad aditiva. Las distribuciones de features entre fraude y no-fraude son casi idénticas.

### 3.3 Generador v2 (`generar_dataset_fraude_v2.py`)

- **Señal fuerte**: PR-AUC ~0.96, AUC-ROC ~0.99
- **Tasa de fraude**: ~15%
- **Mecanismo**: misma estructura que v1, PERO después de asignar `IS_FRAUD=1`, se modifican features post-hoc para reflejar comportamiento fraudulento real.
- **Modificaciones post-hoc en fraudes**:

| Feature | Prob. en fraude | Normal |
|---------|-----------------|--------|
| `operacion_pais` ≠ `customer_country` | 55% | ~10% |
| `dispositivo_reconocido=0` | 60% | ~15% |
| `numero_transacciones_ultima_hora` = 6-20 | 45% | Poisson(2) |
| `importe` > 85% del límite | 40% | ~8% |
| `numero_pin_disponibles=0` | 35% | ~2% |
| `metodo_autenticacion` = firma/3DS | 35% | ~20% |
| `destino_alto_riesgo=1` | 30% | ~12% |
| `tiempo_entre_txns` < 55s | 30% | muy raro |

### 3.4 Columna `IMPACTO_FRAUDE` (multiclase)

Cuando `IS_FRAUD=1`, se asigna impacto según importe:
- `importe < 500` → `IMPACTO_FRAUDE=1` (bajo)
- `500 ≤ importe < 2000` → `IMPACTO_FRAUDE=2` (medio)
- `importe ≥ 2000` → `IMPACTO_FRAUDE=3` (alto)
- `IS_FRAUD=0` → `IMPACTO_FRAUDE=0` (no fraude)

**⚠️ CRÍTICO — Data leakage reparado**: Originalmente los notebooks NO eliminaban la columna objetivo opuesta. Se añadió `drop(columns=other_target)` justo después de cargar los datos en todos los notebooks.

---

## 4. Feature Engineering — v3 (`feature_engineering.py`)

Clase `FeatureEngineer` (hereda de `BaseEstimator, TransformerMixin`). Versión actual: **v3** con **67 features**.

### Cambios principales de v3 respecto a v2:

| Cambio | Detalle |
|--------|---------|
| `estado_cuenta` y `estado_tarjeta` eliminadas | Movidas de CAT_COLS a DROP_COLS por poca señal |
| `numero_fraudes_ultimo_ano` retenido | Pero con capping en 3 → `fraudes_prev_capped` |
| +13 nuevas features | Sesión (5), desviación temporal (3), red (1), flags compuestas (3) |

### Transformaciones completas:

**1. Temporales** (3): `hour`, `day_of_week`, `is_weekday` desde `fecha_hora`

**2. Eliminación** (12): IDs, `geolocalizacion`, `fecha_creacion_tarjeta`, `fecha_hora`, `estado_cuenta`, `estado_tarjeta`

**3. Capping**: `numero_fraudes_ultimo_ano` → `fraudes_prev_capped = min(n, 3)`

**4. Ratios financieros** (11):
- `txn_vs_limit_pct`, `txn_vs_balance_pct`, `txn_vs_monthly_avg_pct`
- `balance_vs_avg_pct`, `outflow_inflow_ratio`, `net_flow_30d`
- `limite_breach_rate`, `txn_intensity`, `balance_utilization`
- `txn_severity`, `tenure_years`

**5. Flags geográficas** (3): `cross_border`, `cross_region`, `same_country_device`

**6. Log transforms** (5): importe, saldo, volúmenes (2), tiempo_última

**7. Codificación categórica** (16): frequency + target encoding para 8 columnas categóricas

**8. Sesión (burst detection)** (5):
- `txn_por_minuto`: txns / (tiempo_última + 1)
- `burst_rapido`: 1 si >5 tx última hora AND última tx < 5 min
- `alta_velocidad`: 1 si >3 tx última hora OR última tx < 30s
- `monto_velocidad`: importe × tx_última_hora
- `tiempo_ultima_bin`: buckets [0-30s, 30s-5m, 5m-1h, 1h+]

**9. Desviación temporal (por cliente)** (3):
- `diff_hora_cliente`: |hora - hora_media del cliente|
- `diff_importe_cliente`: |importe - importe_medio| / importe_medio
- `ratio_actividad_cliente`: tx_última_hora / media_diaria del cliente

**10. Red e interacciones** (4):
- `frecuencia_destino`: count de veces que aparece cada cuenta_destino
- `foreign_unknown_device`: país_extranjero AND dispositivo_no_reconocido
- `night_velocity`: is_night AND alta_velocidad
- `high_ratio_redondeado`: importe/limite > 0.85 AND importe ≈ round(importe)

### Parámetros:
- `encode_target=None`: columna target para target encoding
- `random_state=42`

### ⚠️ Historial de bugs reparados:
1. **`fit()` sin target**: cuando `y=None` pero `self.encode_target` está en X, extrae automáticamente `y = X[self.encode_target]`.
2. **`transform()` sin fecha**: extrae `hour`/`day_of_week` de `fecha_hora` ANTES de dropearlo.
3. **`_safe_ratio()`**: función auxiliar para división segura con NaN handling.

---

## 5. Notebooks

### 5.1 `09_pipeline_completo.ipynb` — Pipeline actual (EL PRINCIPAL)

**Propósito**: pipeline completo actualizado con FE v3, KNNImputer, ensemble calibrado y threshold F2.

**Estructura** (24 celdas):
1. Carga de datos (parametrizado: cambiar `DATASET='v1'` o `DATASET='v2'` en celda 1)
2. Feature Engineering v3 (67 features)
3. Train/Val/Test split 60/20/20 estratificado
4. StandardScaler + KNNImputer (n_neighbors=5)
5. LightGBM con mejores parámetros (diferentes para v1/v2)
6. XGBoost con mejores parámetros
7. Calibración (skipped si diferencia PR-AUC < 0.01)
8. Ensemble: búsqueda de peso óptimo w (maximizando PR-AUC en validación)
9. Threshold F2 sobre validación
10. Evaluación final en test
11. Feature importance

**Cómo usarlo**:
```powershell
cd C:\Dev\NovaPay_ML
# Para v2
jupyter nbconvert --execute --to notebook notebooks\09_pipeline_completo.ipynb
# Para v1, cambiar DATASET = 'v1' en la celda de configuración
```

### 5.2 `07_train_fraud_v1.ipynb` — Entrenamiento v1 (legado)

Entrenamiento con FE v3, LightGBM + XGBoost sin ensemble/KNNImputer/F2. Reemplazado por `09_pipeline_completo.ipynb`.

### 5.3 `08_train_fraud_v2.ipynb` — Entrenamiento v2 (legado)

Misma estructura que `07` pero con datos v2. Reemplazado por `09`.

### 5.4 Notebooks legacy (01–06)

`01`–`06` eran notebooks originales con 8 modelos y GridSearchCV completo. Ya no están en el repositorio (reemplazados por los notebooks en `notebooks/` y los deep-dives en `docs/`).

---

## 6. Modelos Guardados (`.pkl`)

### 6.1 Estructura

Cada `.pkl` contiene el **pipeline completo** en un solo archivo:

```python
{
    'fe':        FeatureEngineer v3 (fitted),
    'scaler':    StandardScaler (fitted, 66 features numéricas),
    'imputer':   KNNImputer (fitted, n_neighbors=5),
    'lgb_model': LightGBM entrenado (67 features, 200 trees),
    'xgb_model': XGBoost entrenado (67 features, 200 trees),
    'best_w':    peso óptimo del ensemble (0.52 v2 / 0.30 v1),
    'best_t':    threshold F2 (0.3223 v2 / 0.3144 v1),
    'best_prec': precisión del threshold en validación,
    'best_rec':  recall del threshold en validación,
    'num_feats': lista de 66 columnas numéricas,
    'metadata': {
        'dataset', 'label', 'fecha', 'n_features', 'n_train/val/test',
        'fraud_rate', 'lightgbm_val_prauc', 'xgboost_val_prauc',
        'ensemble_val_prauc', 'ensemble_test_prauc/auc/precision/recall/f1',
        'best_w', 'f2_threshold', 'calibration_used'
    }
}
```

### 6.2 Archivos

| Archivo | Dataset | PR-AUC test | AUC-ROC test | Precision | Recall | F1 |
|---------|---------|:-----------:|:-----------:|:---------:|:------:|:--:|
| `saved_models/modelo_07_v1.pkl` | v1 (original) | 0.3236 | 0.7135 | 19.7% | 87.2% | 0.3215 |
| `saved_models/modelo_08_v2.pkl` | v2 | 0.9640 | 0.9874 | 77.2% | 95.3% | 0.8529 |

### 6.3 Carga e inferencia

**Para cargar solo necesitas 2 archivos**: el `.pkl` + `feature_engineering.py` en el mismo directorio.
`regenerate_models.py` ya los copia automáticamente a `saved_models/`.

Carga autónoma (funciona copiando la carpeta `saved_models/` a cualquier máquina):

```python
import sys, joblib
from pathlib import Path

pkl_dir = Path('scripts/saved_models')
sys.path.insert(0, str(pkl_dir))
from feature_engineering import FeatureEngineer

obj = joblib.load(str(pkl_dir / 'modelo_08_v2.pkl'))

# Pipeline de inferencia
X = obj['fe'].transform(df_nuevo)
X = X.drop(columns=['IS_FRAUD'], errors='ignore')
X[obj['num_feats']] = obj['scaler'].transform(X[obj['num_feats']])
X[obj['num_feats']] = obj['imputer'].transform(X[obj['num_feats']])

p_lgb = obj['lgb_model'].predict_proba(X)[:, 1]
p_xgb = obj['xgb_model'].predict_proba(X)[:, 1]
y_prob = obj['best_w'] * p_lgb + (1 - obj['best_w']) * p_xgb
y_pred = (y_prob >= obj['best_t']).astype(int)
```

Ver `scripts/inference_example.py` para ejemplo completo.

Ejecutar desde `scripts/`:

```powershell
cd C:\Dev\NovaPay_ML
python scripts\regenerate_models.py
```

El script entrena ambos modelos (v1 y v2) con pipeline completo (FE v3 → KNNImputer → Scaler → LGB + XGB → ensemble → F2 thr) y los guarda.

---

## 7. Decisiones Técnicas Clave

### 7.1 sklearn 1.8 breaking changes

| Cambio | Fix |
|--------|-----|
| `multi_class='multinomial'` eliminado de LogisticRegression | Eliminar el parámetro |
| `cv='prefit'` eliminado de `CalibratedClassifierCV` | Usar `cv=3` o saltar calibración |
| Argumentos posicionales prohibidos | `GradientBoostingClassifier(200) → n_estimators=200` |

### 7.2 Por qué NO se usa oversampling

Datos sintéticos (Faker). Oversampling duplicaría patrones sintéticos → overfitting.
Se usa **cost-sensitive learning** + **optimización de threshold F2**.

### 7.3 Por qué F2

F2 (beta=2) pondera recall 2× más que precisión. Para v2, precision ≥ 60% es alcanzable
(con F2 se obtiene ~77% precisión y ~95% recall). Para v1, precision ≥ 60% da recall inútil
(~2.6%), así que F2 da ~20% precision con ~87% recall.

### 7.4 Calibración

Se evalúa pero se salta si diferencia PR-AUC < 0.01 (nunca mejora significativamente).
sklearn 1.8 eliminó `cv='prefit'`, se usa `cv=3`.

### 7.5 Data leakage cruzado

`IMPACTO_FRAUDE` y `IS_FRAUD` correlacionados por construcción. Solución:
`df.drop(columns=[other_target])` al cargar datos.

### 7.6 `estado_cuenta` y `estado_tarjeta` excluidas

Eliminadas de features por poca señal discriminativa y riesgo de leakage (estado de cuenta
puede depender del fraude posterior).

### 7.7 Solo ensemble LGB + XGB

De 8 modelos iniciales, se redujo a los 2 más competitivos para datos tabulares con
clase desbalanceada. El ensemble ponderado supera consistentemente a cada modelo individual.

---

## 8. Estado Actual

### Completado:
- Feature Engineering v3 con 67 features (sesión, red, desviaciones, capping, flags compuestas)
- `09_pipeline_completo.ipynb` verificado (v2: PR-AUC 0.9640, Recall 95.3%, Precision 77.2%)
- Modelos `.pkl` regenerados con pipeline completo (FE v3 + Scaler + Imputer + Ensemble + Threshold)
- `feature_engineering.py` se copia automáticamente a `saved_models/` para carga autónoma
- `inference_example.py`: ejemplo funcional de inferencia (ahora usa `sys.path.insert(0, pkl_dir)`)
- `prediccion_lote.py`: batch inference — CSV de entrada → CSV con predicciones añadidas
- `evaluacion_rondas.py`: evalúa modelo en rondas de 100 txns, simula producción, detecta drift (PSI)
- `generar_muestra_sin_etiqueta.py`: genera CSV+JSON sin IS_FRAUD con perfiles de riesgo (mixto/sospechoso/fraude)
- `residuos_y_evaluacion.md`: estudio completo de técnicas de análisis de residuos
- `regenerate_models.py`: script que regenera ambos modelos
- `app.py`: API FastAPI para inferencia en producción (v5.0.0 con validación Pydantic v2)
- Todos los notebooks y scripts con paths correctos (`scripts/`, `notebooks/`, `data/`, `docs/`)
- `IMPACTO_FRAUDE` eliminado de generadores — se calcula post-inferencia como regla de negocio
- Generadores actualizados para guardar en `data/` automáticamente
- Estructura final del proyecto consolidada (data/ scripts/ notebooks/ docs/)
- `03_feature_engineering_deep_dive.ipynb` actualizado con sección v3
- `06_model_selection_deep_dive.ipynb` actualizado con sección ensemble
- Deep-dive notebooks ejecutándose sin errores

### Pendiente / A decidir (próximas iteraciones):
- Refinamiento continuo del modelo: hiperparámetros, nuevas features (v4), otros algoritmos
- Mejora en la generación de datos sintéticos
- Llevar a producción: API, monitoreo, métricas en vivo
- CatBoost no instalado — notebooks lo manejan con try/except
- Para producción real, el umbral F2 debe recalibrarse con datos reales
- `modelo_07_v1.pkl` tiene PR-AUC 0.3236 — dataset con señal débil, no usar en producción

### Roles:
- **Colaborador (ML/DS)**: entrenamiento del modelo, feature engineering, generación de datos sintéticos
- No hay dependencias externas bloqueantes para continuar

---

### Pydantic v2 — Validación de entrada en API

`app.py` usa `field_validator` y `ConfigDict` de Pydantic v2 para validar
datos antes de que el modelo los procese:

```python
class Transaccion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1)
    edad_cliente: int = Field(..., ge=0, le=120)

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
```

Campos validados: `importe_transaccion` (>0), `dispositivo_reconocido` (0/1),
`is_night` (0/1), `is_weekend` (0/1), `destino_alto_riesgo` (0/1),
`numero_pin_disponibles` (≥0), `edad_cliente` (0-120).
La respuesta también valida `prob_fraud` (0.0-1.0) para evitar errores numéricos.

Sin riesgo de ruptura: los validadores solo rechazan valores imposibles (edad 999,
importe negativo, flag=3), nada que antes fuera válido.

---

## 9. Cómo Usar

Todos los comandos se ejecutan desde la raíz del proyecto:

```powershell
cd C:\Dev\NovaPay_ML
```

### 9.1 Entrenar / regenerar modelos

```powershell
python scripts\regenerate_models.py
```

Entrena v1 y v2 desde cero con pipeline completo (FE v3 → KNNImputer → Scaler → LGB + XGB → ensemble → F2 thr) y copia `feature_engineering.py` a `saved_models/` para carga autónoma.

### 9.2 Generar datos sintéticos

```powershell
# Dataset original v1 (señal débil, 10K transacciones)
python scripts\synthetic_data\generar_dataset_fraude.py

# Dataset v2 (señal fuerte, 10K transacciones)
python scripts/synthetic_data/generar_dataset_fraude_v2.py

# Dataset sin etiqueta (perfil mixto, 200 tx)
python scripts/generar_muestra_sin_etiqueta.py

# Dataset con perfil fraudulento
python scripts/generar_muestra_sin_etiqueta.py --perfil fraude

# Generar los 3 perfiles a la vez
python scripts/generar_muestra_sin_etiqueta.py --perfil todo
```

Los datasets etiquetados se guardan en `data/`. El dataset sin etiqueta se guarda donde se indique.

### 9.3 Inferencia (una transacción)

```powershell
# Ejemplo completo con modelo v2
python scripts\inference_example.py

# O vía API
uvicorn app:app --reload
# POST http://localhost:8000/predict
```

### 9.4 Batch inference (CSV → CSV con predicciones + FE)

```powershell
# Genera CSV con columnas originales + 45 FE features + predicciones
python scripts\prediccion_lote.py --input data/muestra_test.csv --output data/predicciones.csv --modelo v2

# Sin columna impacto_fraude (opcional)
python scripts\prediccion_lote.py --input data/muestra_test.csv --output data/predicciones.csv --modelo v2 --no-impacto
```

### 9.5 Separar predicciones en JSON

```powershell
# Paso 1: batch inference
python scripts/prediccion_lote.py --input data/dataset_fraude_v2.csv --output data/predicciones_v2.csv --modelo v2

# Paso 2: separar en dos JSON
python scripts\separar_prediccion_json.py --input data/predicciones_v2.csv --output-dir data/
```

Esto genera:
- `data/transacciones.json` — solo columnas originales
- `data/predicciones.json` — id_transaccion + 45 features FE + predicciones

### 9.6 Aplanar predicciones a formato compacto

```powershell
python scripts\aplanar_predicciones.py --input data/predicciones.json --output data/predicciones_aplanadas.json
```

Reduce cada registro a 10 campos clave: `id_transaccion`, `is_fraud`, `prob_fraud`, `impacto_fraude`, `es_transfronteriza`, `ratio_imp_limite`, `intensidad_tx`, `severidad_tx`, `flujo_neto_30d`, `mensaje`.

### 9.7 Evaluación por rondas (simular producción y drift)

```powershell
# Baseline — distribución estable
python scripts\evaluacion_rondas.py --modelo v2 --rondas 50 --drift baseline

# Drift suave — cambio gradual en features (rondas 20-40)
python scripts\evaluacion_rondas.py --modelo v2 --rondas 50 --drift suave

# Drift abrupto — cambio brusco en ronda 30
python scripts\evaluacion_rondas.py --modelo v2 --rondas 50 --drift abrupto

# Concept drift — nuevo patrón de fraude no visto (ronda 25)
python scripts\evaluacion_rondas.py --modelo v2 --rondas 50 --drift concepto

# Guardar resultados a CSV
python scripts\evaluacion_rondas.py --modelo v2 --rondas 100 --drift suave --output data/metricas_rondas.csv
```

### 9.8 Carga autónoma del modelo (solo 2 archivos)

```python
import sys, joblib
from pathlib import Path

pkl_dir = Path('scripts/saved_models')
sys.path.insert(0, str(pkl_dir))
from feature_engineering import FeatureEngineer

obj = joblib.load(str(pkl_dir / 'modelo_08_v2.pkl'))

# Pipeline completo disponible:
# obj['fe'], obj['scaler'], obj['imputer'],
# obj['lgb_model'], obj['xgb_model'],
# obj['best_w'], obj['best_t'], obj['num_feats']
```

### 9.9 Explorar resultados

```powershell
# Pipeline completo (interactivo)
jupyter notebook notebooks\09_pipeline_completo.ipynb

# Feature engineering deep dive
jupyter notebook docs\03_feature_engineering_deep_dive.ipynb

# Model selection deep dive
jupyter notebook docs\06_model_selection_deep_dive.ipynb

# Análisis de residuos
notepad docs\residuos_y_evaluacion.md
```

### Packages principales:
- Python 3.14, scikit-learn 1.8.0, lightgbm 4.6.0, xgboost 3.2.0
- pandas 2.3.3, numpy 2.4.3, joblib 1.5.3
- Faker 40.15.0, matplotlib 3.10.8, seaborn 0.13.2, plotly 6.6.0

---

## 10. Glosario

| Término | Significado |
|---------|-------------|
| OOF | Out-Of-Fold: predicciones en datos no vistos durante training usando CV |
| F2 | F-beta score con beta=2 (recall pesa 2× más que precisión) |
| PR-AUC | Area Under Precision-Recall Curve (métrica principal para clases desbalanceadas) |
| AUC-ROC | Area Under ROC Curve |
| Cost-sensitive | Ajustar pesos de clase en lugar de oversampling |
| Target encoding | Codificar categóricas con la media del target por categoría |
| Leakage | Información del futuro/fuera de muestra que "filtra" hacia features |
| Burst detection | Detección de ráfagas de transacciones en corto tiempo |
| KNNImputer | Imputación de valores faltantes basada en k vecinos más cercanos |
