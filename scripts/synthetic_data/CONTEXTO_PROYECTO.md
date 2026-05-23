# Proyecto NovaPay ML — Documento de Contexto Completo

Este documento describe la totalidad del proyecto de detección de fraude en transacciones
financieras de NovaPay. Está escrito para que otro asistente (LLM) pueda retomar el trabajo
sin necesidad de preguntar al usuario.

---

## 1. Propósito del Proyecto

Construir un pipeline de machine learning para detectar **fraude en transacciones bancarias**
usando datos sintéticos generados con Faker. El proyecto incluye:

- Generación de datos sintéticos transaccionales multi-nivel (clientes, cuentas, tarjetas, transacciones)
- Feature engineering con codificación target y frecuencias
- Entrenamiento y comparación de múltiples modelos (LightGBM, XGBoost, Random Forest, Logistic Regression, etc.)
- Optimización de hiperparámetros con GridSearchCV
- Selección de threshold optimizando F2 (recall-first con restricción de precisión)
- Pipeline de producción para inferencia
- Documentación de resultados

**Entorno**: Windows, Python 3.14, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0

---

## 2. Estructura del Proyecto

```
C:\Dev\NovaPay_ML\
├── model/                          # Notebooks y código principal
│   ├── feature_engineering.py      # Transformer personalizado sklearn
│   ├── __init__.py                 # Package init
│   ├── 01_train_fraud.ipynb        # Entrenamiento principal IS_FRAUD (binario)
│   ├── 02_train_impacto.ipynb      # Entrenamiento IMPACTO_FRAUDE (multiclase)
│   ├── 03_feature_engineering_deep_dive.ipynb  # Análisis exploratorio
│   ├── 04_pipeline_fraud.ipynb     # Pipeline producción IS_FRAUD
│   ├── 05_pipeline_impacto.ipynb   # Pipeline producción IMPACTO_FRAUDE
│   ├── 06_model_selection_deep_dive.ipynb  # Comparación profunda de modelos
│   └── 07_train_fraud_rigorous.ipynb  # Notebook riguroso (2 modelos, train/val/test)
│
├── scripts/
│   └── synthetic_data/
│       ├── generar_dataset_fraude.py          # Generador v1 (señal débil)
│       ├── generar_dataset_fraude_mejorado.py # Generador v2 (señal fuerte)
│       ├── mejora_senal_fraude.md             # Documentación de mejora de señal
│       ├── esquema_bd.sql                     # Esquema de base de datos
│       └── sdv/
│           └── generar_dataset_sdv.py         # Generador alternativo con SDV
│
└── Notebooks/
    └── EDA.ipynb                               # Análisis exploratorio inicial
```

---

## 3. Datos Sintéticos

### 3.1 Dataset generado

Genera **10,000 transacciones** con estructura jerárquica:

| Nivel | Entidad | Cantidad | Columnas clave |
|-------|---------|----------|----------------|
| 1 | Cliente | 2,000 | `tipo_cliente`, `edad_cliente`, `customer_country`, `tenure`, `importe_medio_mensual` |
| 2 | Cuenta | ~4,500 | `estado_cuenta`, `saldo_actual`, `volumen_entrante/saliente` |
| 3 | Tarjeta | ~9,000 | `estado_tarjeta`, `antiguedad_tarjeta_dias`, `limite_importe_transacciones` |
| 4 | Transacción | 10,000 | `importe_transaccion`, `is_night`, `dispositivo_reconocido`, `destino_alto_riesgo` |

**Columnas del dataset** (45 columnas, mismas en v1 y v2):

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

- **Señal débil**: PR-AUC ~0.35, AUC-ROC ~0.73
- **Tasa de fraude**: ~15%
- **Mecanismo**: probabilidad aditiva con 15+ factores, cada uno contribuye 2-25%. Las features se generan independientemente con la misma distribución para todas las transacciones.
- **Problema**: la única dependencia features → label es la fórmula de probabilidad aditiva. Las distribuciones de features entre fraude y no-fraude son casi idénticas.

### 3.3 Generador v2 (`generar_dataset_fraude_mejorado.py`)

- **Señal fuerte**: PR-AUC ~0.96, AUC-ROC ~0.99
- **Tasa de fraude**: ~15%
- **Mecanismo**: misma estructura que v1, PERO después de asignar `IS_FRAUD=1` (usando la misma probabilidad base de v1), se modifican features post-hoc para reflejar comportamiento fraudulento real. Esto crea **distribuciones diferentes** entre clases.
- **Modificaciones post-hoc en fraudes**:

| Feature | Prob. en fraude | Distribución normal |
|---------|-----------------|-------------------|
| `operacion_pais` ≠ `customer_country` | 55% | ~10% |
| `dispositivo_reconocido=0` | 60% | ~15% |
| `estado_tarjeta` robada/extraviada | 35% | ~7% |
| `numero_transacciones_ultima_hora` = 6-20 | 45% | Poisson(2) |
| `importe` > 85% del límite | 40% | ~8% |
| `numero_pin_disponibles=0` | 35% | ~2% |
| `metodo_autenticacion` = firma/3DS | 35% | ~20% |
| `destino_alto_riesgo=1` | 30% | ~12% |
| `tiempo_desde_ultima_transaccion` < 55s | 30% | muy raro |

### 3.4 Columna `IMPACTO_FRAUDE` (multiclase)

Cuando `IS_FRAUD=1`, se asigna impacto según importe:
- `importe < 500` → `IMPACTO_FRAUDE=1` (bajo)
- `500 ≤ importe < 2000` → `IMPACTO_FRAUDE=2` (medio)
- `importe ≥ 2000` → `IMPACTO_FRAUDE=3` (alto)
- `IS_FRAUD=0` → `IMPACTO_FRAUDE=0` (no fraude)

**⚠️ CRÍTICO — Data leakage reparado**: Originalmente los notebooks NO eliminaban la columna objetivo opuesta. Cuando se predecía `IS_FRAUD`, la columna `IMPACTO_FRAUDE` quedaba en X. Como `IMPACTO_FRAUDE` solo es distinto de cero cuando hay fraude, esto filtraba la etiqueta. Se añadió `drop(columns=other_target)` justo después de cargar los datos en todos los 6 notebooks.

---

## 4. Feature Engineering (`feature_engineering.py`)

Clase `FeatureEngineer` (hereda de `BaseEstimator, TransformerMixin`) para usar en pipelines de sklearn.

### Transformaciones:

1. **Temporales**: extrae `hour`, `day_of_week` de `fecha_hora`, luego elimina `fecha_hora`
2. **Eliminación**: `DROP_COLS` (IDs, geolocalización, `numero_fraudes_ultimo_ano`)
3. **Ratio features** (9):
   - `txn_vs_limit_pct` = importe / límite
   - `txn_vs_balance_pct` = importe / saldo
   - `txn_vs_monthly_avg_pct` = importe / media mensual
   - `balance_vs_avg_pct` = saldo / saldo medio 30d
   - `outflow_inflow_ratio` = saliente / entrante
   - `net_flow_30d` = entrante - saliente
   - `limite_breach_rate` = superaciones / 7
   - `txn_intensity` = tx_última_hora / (tiempo_última + 1)
   - `balance_utilization` = saldo_medio / (límite × 10 + 1)
   - `txn_severity` = importe × tx_última_hora
   - `tenure_years` = tenure / 365
4. **Flags geográficas**: `cross_border`, `cross_region`, `same_country_device`
5. **Log transforms**: importe, saldo, volúmenes, tiempo_última
6. **Codificación categórica**: frequency encoding + target encoding (si se especifica `encode_target`)
7. **Salida**: ~45 columnas numéricas

### Parámetros:
- `encode_target=None`: columna target para target encoding
- `n_folds=5`: folds para target encoding (no implementado, siempre usa full)
- `random_state=42`

### ⚠️ Historial de bugs reparados:

1. **`fit()` sin target**: cuando `y=None` pero `self.encode_target` está en X, extrae automáticamente `y = X[self.encode_target]`. Antes devolvía error.
2. **`transform()` sin fecha**: el código extrae `hour`/`day_of_week` de `fecha_hora` ANTES de dropearlo. Antes el drop ocurría primero.
3. **`_safe_ratio()`**: función auxiliar para división segura con NaN handling.

---

## 5. Notebooks

### 5.1 `01_train_fraud.ipynb` — Entrenamiento principal IS_FRAUD

- **Propósito**: pipeline completo para clasificación binaria de fraude
- **Modelos** (8): LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost (condicional), ExtraTrees, AdaBoost
- **GridSearchCV** con scoring `average_precision` (cambiado de `roc_auc` por ser más apropiado para clases desbalanceadas)
- **Selección de threshold**: OOF (out-of-fold) con optimización F2
  - Recorre 500 thresholds en `np.linspace(0.01, 0.99, 500)`
  - Para cada threshold, calcula F2
  - Selecciona threshold con **máximo F2** (sobre OOF, no sobre train)
  - Aplica threshold en test
- **Calibración**: `CalibratedClassifierCV` (con `cv=5` después de que sklearn 1.8 eliminó `cv='prefit'`)
- **F2 scoring**: usa `fbeta_score(beta=2)` como métrica de decisión
- **Salida**: mejores parámetros, threshold óptimo, matriz de confusión, feature importance

### 5.2 `02_train_impacto.ipynb` — Entrenamiento IMPACTO_FRAUDE (multiclase)

- **Propósito**: predecir el nivel de impacto del fraude (0=no fraude, 1=bajo, 2=medio, 3=alto)
- **Solo transacciones donde `IMPACTO_FRAUDE > 0`** (el modelo de impacto solo ve fraudes)
- **Modelos**: RandomForest, XGBoost, LightGBM (multiclase)
- **GridSearchCV** scoring `f1_weighted`
- **Nota**: `IS_FRAUD` eliminado de features para evitar leakage

### 5.3 `03_feature_engineering_deep_dive.ipynb` — Análisis EDA

- **Propósito**: entender distribución de features, correlaciones, separación entre clases
- **Correcciones**: `GradientBoostingClassifier(200, ...)` → `n_estimators=200` (sklearn 1.8 eliminó argumentos posicionales), import `_safe_ratio`, `StandardScaler`

### 5.4 `04_pipeline_fraud.ipynb` — Pipeline producción IS_FRAUD

- **Propósito**: pipeline reutilizable con `Pipeline` de sklearn (FeatureEngineer + modelo)
- **Mismos modelos y approach que `01` pero empaquetado como pipeline**
- **Serialización con joblib**
- **Selección de threshold**: igual que `01` (F2 sobre OOF)

### 5.5 `05_pipeline_impacto.ipynb` — Pipeline producción IMPACTO_FRAUDE

- **Propósito**: pipeline multiclase para producción
- **Misma estructura que `04` pero con target multiclase y solo fraudes**

### 5.6 `06_model_selection_deep_dive.ipynb` — Comparación profunda

- **Propósito**: comparación estadística de modelos con CV repetido
- **Análisis de curva PR, ROC, lift charts**
- **Selección de threshold con F2 y análisis de costo-beneficio**
- **Correcciones**: mismas que `03`

### 5.7 `07_train_fraud_rigorous.ipynb` — Notebook riguroso (el principal)

**Propósito**: entrenamiento riguroso con solo los 2 modelos más competitivos y separación train/validation/test.

**Estructura** (25 celdas de código):
1. Imports, carga de datos, `drop(columns=other_target)` para evitar leakage
2. `FeatureEngineer` → train/val/test split (60/20/20 estratificado)
3. LightGBM GridSearchCV con 288 combinaciones
4. XGBoost GridSearchCV con 288 combinaciones
5. Comparación en validación: PR-AUC, AUC-ROC, Precision@Recall=90%
6. Calibración (skipped si diferencia PR-AUC < 0.01)
7. Selección de threshold:
   - Busca threshold que maximice F2 en validación
   - Si precision ≥ 60% es alcanzable con recall < 10%, usa F2 automáticamente
   - Muestra comparativa F1 vs F2
8. Evaluación final en test con threshold seleccionado
9. Reporte completo: matriz de confusión, PR-AUC, AUC-ROC, F1, Precision@Recall=90%, Alertas/100k
10. Feature importance, conclusiones

**Resultados con datos v2**:
| Métrica | Con threshold F2 |
|---------|-----------------|
| Modelo seleccionado | LightGBM |
| PR-AUC (test) | ~0.96 |
| AUC-ROC (test) | ~0.99 |
| Precision | ~86% |
| Recall | ~90% |
| F1 (fraude) | ~0.88 |

**Tiempo de ejecución**: ~26 minutos (ambos grids de 288 combinaciones)

---

## 6. Decisiones Técnicas Clave

### 6.1 sklearn 1.8 breaking changes

| Cambio | Notebooks afectados | Fix |
|--------|-------------------|-----|
| `multi_class='multinomial'` eliminado de LogisticRegression | `01`, `04` | Eliminar el parámetro |
| `cv='prefit'` eliminado de `CalibratedClassifierCV` | `01`, `07` | Usar `cv=5` o saltar calibración |
| Argumentos posicionales prohibidos en constructores | `03`, `06` | `GradientBoostingClassifier(200) → n_estimators=200` |

### 6.2 Por qué NO se usa oversampling

Los datos son sintéticos (Faker). Oversampling (SMOTE, ADASYN, etc.) duplicaría patrones
sintéticos existentes, lo que lleva a overfitting. Se usa **cost-sensitive learning**
(`scale_pos_weight`/`class_weight`) + **optimización de threshold**.

### 6.3 Por qué F2 en lugar de precisión ≥ 60%

El threshold con precisión ≥ 60% es técnicamente alcanzable (~63% precisión), pero el
recall es solo ~4% (con datos v1). Esto es inútil operacionalmente. F2 (beta=2) pondera
recall 2× más que precisión, dando un threshold más práctico (~26-86% precisión,
~79-91% recall dependiendo de la versión de datos).

### 6.4 Calibración

Se evalúa si la calibración mejora PR-AUC. Si la diferencia es < 0.01, se salta
(sklearn 1.8 eliminó `cv='prefit'`). En la práctica, la calibración no mejora el
ranking (PR-AUC), solo ajusta las probabilidades absolutas.

### 6.5 Data leakage cruzado

`IMPACTO_FRAUDE` y `IS_FRAUD` están correlacionados por construcción (IMPACTO=0
cuando no hay fraude). Si uno queda como feature al predecir el otro, hay leakage.
Solución: eliminar la columna opuesta explícitamente en cada notebook.

---

## 7. Estado Actual (última sesión)

### Completado:
- Todos los notebooks corren sin errores en sklearn 1.8
- Data leakage reparado en los 6 notebooks
- `feature_engineering.py` reparado (target encoding, temporal features)
- Grillas de hiperparámetros con `class_weight`/`scale_pos_weight`
- F2 scoring y selección de threshold OOF
- Notebook riguroso `07` creado y verificado
- Generador v2 creado (`generar_dataset_fraude_mejorado.py`) con señal fuerte
- Documentación `mejora_senal_fraude.md`

### Pendiente / A decidir:
- El notebook `07` está configurado para usar datos v1. Para usar datos v2, cambiar
  la ruta del CSV a `dataset_fraude_mejorado.csv` y re-ejecutar.
- CatBoost no está instalado (`catboost` package not found) — los notebooks lo
  manejan con try/except.
- Los resultados documentados arriba (PR-AUC 0.96, Recall 90%) son con datos v2.
  Con datos v1 los resultados son menores (PR-AUC ~0.34-0.36).

---

## 8. Cómo Continuar

### Para entrenar con datos mejorados (v2):

```powershell
cd C:\Dev\NovaPay_ML\scripts\synthetic_data
python generar_dataset_fraude_mejorado.py
Copy-Item dataset_fraude_mejorado.csv ..\..\model\dataset_fraude.csv -Force
cd ..\..\model
jupyter nbconvert --execute --to notebook 07_train_fraud_rigorous.ipynb
```

### Para ejecutar un notebook específico:

```powershell
cd C:\Dev\NovaPay_ML\model
jupyter nbconvert --execute --to notebook 01_train_fraud.ipynb --ExecutePreprocessor.timeout=3600
```

### Packages principales:
- Python 3.14
- scikit-learn 1.8.0
- lightgbm 4.6.0
- xgboost 3.2.0
- pandas 2.3.3
- numpy 2.4.3
- Faker 40.15.0
- matplotlib 3.10.8
- seaborn 0.13.2
- plotly 6.6.0
- joblib 1.5.3

---

## 9. Glosario

| Término | Significado |
|---------|-------------|
| OOF | Out-Of-Fold: predicciones en datos no vistos durante training usando CV |
| F2 | F-beta score con beta=2 (recall pesa 2× más que precisión) |
| PR-AUC | Area Under Precision-Recall Curve (métrica principal para clases desbalanceadas) |
| AUC-ROC | Area Under ROC Curve |
| Cost-sensitive | Ajustar pesos de clase en lugar de oversampling |
| Target encoding | Codificar categóricas con la media del target por categoría |
| Leakage | Información del futuro/fuera de muestra que "filtra" hacia features |
| Alertas/100k | Número de alertas (predicciones positivas) por cada 100,000 transacciones |
