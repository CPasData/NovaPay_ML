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
├── model/                          # Código principal y modelos
│   ├── feature_engineering.py      # Transformer v3 (67 features)
│   ├── __init__.py                 # Package init
│   ├── 01_train_fraud.ipynb        # Entrenamiento IS_FRAUD (8 modelos, legado)
│   ├── 02_train_impacto.ipynb      # Entrenamiento IMPACTO_FRAUDE (multiclase, legado)
│   ├── 03_feature_engineering_deep_dive.ipynb  # Análisis v3 (incluye sección sesión/red)
│   ├── 04_pipeline_fraud.ipynb     # Pipeline producción IS_FRAUD (legado)
│   ├── 05_pipeline_impacto.ipynb   # Pipeline producción IMPACTO_FRAUDE (legado)
│   ├── 06_model_selection_deep_dive.ipynb  # Comparación de modelos + ensemble
│   ├── 07_train_fraud_rigorous.ipynb  # Notebook riguroso v1 (LGB + XGB, legado)
│   ├── 08_train_fraud_mejorado.ipynb  # Notebook riguroso v2 (LGB + XGB, legado)
│   ├── 09_pipeline_completo.ipynb  # Pipeline completo actualizado v3
│   ├── regenerate_models.py        # Script que regenera ambos .pkl con pipeline completo
│   ├── inference_example.py        # Ejemplo de inferencia con modelo guardado
│   ├── save_models.py              # Script legacy (reemplazado por regenerate_models.py)
│   └── saved_models/
│       ├── modelo_07_v1.pkl        # Pipeline completo entrenado con v1
│       └── modelo_08_v2.pkl        # Pipeline completo entrenado con v2
│
├── scripts/
│   └── synthetic_data/
│       ├── generar_dataset_fraude.py          # Generador v1 (señal débil)
│       ├── generar_dataset_fraude_mejorado.py # Generador v2 (señal fuerte)
│       ├── mejora_senal_fraude.md             # Documentación de mejora de señal
│       ├── CONTEXTO_PROYECTO.md               # ← ESTE DOCUMENTO
│       ├── esquema_bd.sql                     # Esquema de base de datos
│       └── sdv/
│           └── generar_dataset_sdv.py         # Generador alternativo con SDV
│
└── Notebooks/
    └── EDA.ipynz                               # Análisis exploratorio inicial
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

### 3.3 Generador v2 (`generar_dataset_fraude_mejorado.py`)

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
# Para v2 (mejorado)
jupyter nbconvert --execute --to notebook model\09_pipeline_completo.ipynb
# Para v1, cambiar DATASET = 'v1' en la celda de configuración
```

### 5.2 `07_train_fraud_rigorous.ipynb` — Notebook riguroso (legado, v1)

Idéntico en estructura a `09` pero sin KNNImputer, sin ensemble, sin FE v3. Reemplazado por `09`.

### 5.3 `08_train_fraud_mejorado.ipynb` — Notebook mejorado (legado, v2)

Misma estructura que `07` pero con datos v2. Reemplazado por `09`.

### 5.4 Notebooks legacy (01–06)

`01`–`06` son notebooks originales con 8 modelos, GridSearchCV completo. No se actualizaron a v3.
Ejecutan sin errores en sklearn 1.8 pero usan FE v1/v2.

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
| `saved_models/modelo_08_v2.pkl` | v2 (mejorado) | 0.9640 | 0.9874 | 77.2% | 95.3% | 0.8529 |

### 6.3 Carga e inferencia

**Solo se necesita 1 archivo** (el `.pkl`) + `feature_engineering.py` accesible en el `sys.path`:

```python
import sys, joblib
sys.path.append('/ruta/al/proyecto')
from model.feature_engineering import FeatureEngineer  # necesaria para deserializar

obj = joblib.load('modelo_08_v2.pkl')

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

Ver `model/inference_example.py` para ejemplo completo.

### 6.4 Regeneración

Ejecutar desde `model/`:

```powershell
cd C:\Dev\NovaPay_ML\model
python regenerate_models.py
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
- `inference_example.py`: ejemplo funcional de inferencia
- `regenerate_models.py`: script que regenera ambos modelos
- Todos los notebooks actualizados con paths correctos (`Path.cwd()` en lugar de `Path.cwd().parent`)
- `03_feature_engineering_deep_dive.ipynb` actualizado con sección v3
- `06_model_selection_deep_dive.ipynb` actualizado con sección ensemble
- Deep-dive notebooks ejecutándose sin errores

### Pendiente / A decidir:
- CatBoost no instalado — notebooks lo manejan con try/except
- Para producción real, el umbral F2 debe recalibrarse con datos reales
- `modelo_07_v1.pkl` tiene PR-AUC 0.3236 — señalar que es un dataset con señal débil, no usar en producción

---

## 9. Cómo Usar

### Para entrenar (regenerar modelos):

```powershell
cd C:\Dev\NovaPay_ML\model
python regenerate_models.py
```

### Para inferencia en producción:

```powershell
cd C:\Dev\NovaPay_ML
python model\inference_example.py
```

### Para explorar resultados:

```powershell
cd C:\Dev\NovaPay_ML
# Pipeline completo (interactivo)
jupyter notebook model\09_pipeline_completo.ipynb
# Feature engineering deep dive
jupyter notebook model\03_feature_engineering_deep_dive.ipynb
# Model selection deep dive
jupyter notebook model\06_model_selection_deep_dive.ipynb
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
