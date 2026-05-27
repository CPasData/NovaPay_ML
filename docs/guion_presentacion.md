# NovaPay ML — Guión de Presentación del Proyecto
## Operación Centinela — Detección de Fraude Transaccional

---

## Índice

1. [Apertura — El Problema de Negocio](#1-apertura--el-problema-de-negocio)
2. [Visión General del Sistema](#2-visión-general-del-sistema)
3. [Evolución: v1 → v2 → v3](#3-evolución-v1--v2--v3)
4. [Deep Dive Técnico](#4-deep-dive-técnico)
5. [Métricas y Resultados](#5-métricas-y-resultados)
6. [Arquitectura de Producción](#6-arquitectura-de-producción)
7. [Preguntas Frecuentes y Respuestas](#7-preguntas-frecuentes-y-respuestas)

---

## 1. Apertura — El Problema de Negocio

### Guión (2-3 min)

"Buenos días a todos. Hoy voy a presentar el proyecto NovaPay ML — Operación Centinela.
Se trata de un sistema de detección de fraude en transacciones financieras que hemos desarrollado
desde cero.

**El problema**: NovaPay procesa ~200.000 transacciones al día. De ellas, aproximadamente un 3%
son fraudulentas. Con un equipo de analistas que puede revisar ~200 alertas al día, necesitamos
un sistema que:

1. Detecte fraudes con alta precisión (mínimo 75% para no saturar al equipo)
2. Recupere el máximo de fraudes posible (F2-score prioriza recall sobre precisión)
3. Opere en tiempo real (cada transacción se decide en <100ms)
4. Se adapte a cambios en los patrones de fraude (concept drift)

Hemos pasado por 3 iteraciones de modelo (v1, v2, v3), mejorando progresivamente tanto
la señal sintética como la arquitectura. Hoy operamos con el modelo v3, que tasa de fraude
realista del 3% y umbrales específicos por canal de pago."

---

## 2. Visión General del Sistema

### Para negocio (1 min)

"El sistema funciona como un **interceptor** que recibe cada transacción, la evalúa con un modelo
de machine learning, y devuelve una decisión en milisegundos. El output son **10 campos** que
permiten al analista priorizar su trabajo:

| Campo | Qué significa | Para qué sirve |
|---|---|---|
| `is_fraud` | Decisión binaria (0/1) | Activar bloqueo |
| `prob_fraud` | Probabilidad 0-100% | Ordenar alertas |
| `impacto_fraude` | Bajo/Medio/Alto | Priorizar por importe |
| `es_transfronteriza` | Misma/otro país | Señal de riesgo adicional |
| `intensidad_tx` | Velocidad transaccional | Detectar automatización |
| `flujo_neto_30d` | Déficit financiero | Perfil de liquidez |

El flujo completo: una transacción entra → pasa por Feature Engineering → escalado →
imputación → ensemble LightGBM+XGBoost → threshold por canal → respuesta con 10 campos."

### Diagrama conceptual

```
POST /predict
     ↓
┌─ Feature Engineering ──────────────┐
│  44 columnas crudas → 69 features   │
└──────────┬─────────────────────────┘
           ↓
┌─ Scaler (StandardScaler) ──────────┐
│  Normaliza features numéricas       │
└──────────┬─────────────────────────┘
           ↓
┌─ Imputer (KNNImputer, n=5) ────────┐
│  Rellena nulos con vecinos          │
└──────────┬─────────────────────────┘
           ↓
┌─ Ensemble ─────────────────────────┐
│  LightGBM (65%) + XGBoost (35%)    │
│  → probabilidad de fraude           │
└──────────┬─────────────────────────┘
           ↓
┌─ Threshold por canal ──────────────┐
│  tarjeta: 0.759                     │
│  transferencia: 0.724              │
│  bizum: 0.783                      │
└──────────┬─────────────────────────┘
           ↓
┌─ Respuesta (10 campos) ────────────┐
│  is_fraud, prob_fraud, impacto...  │
└────────────────────────────────────┘
```

---

## 3. Evolución: v1 → v2 → v3

### Lo que más destacaría

| Aspecto | v1 | v2 | v3 | Por qué importa |
|---|---|---|---|---|
| **Tasa de fraude** | ~15% | ~15% | **3%** | Realista: el fraude real es <5% |
| **Registros** | 10.000 | 10.000 | **100.000** | +10× datos = modelo más robusto |
| **Canales** | tarjeta, transferencia | tarjeta, transferencia | + **bizum** | Canal real en crecimiento |
| **Features** | 67 | 67 | **69** | z-score real, eliminación de ruido |
| **PR-AUC** | 0.35 → 0.96 | 0.96 | **0.90** | Mantiene con 5× menos fraude |
| **Recall@k** | — | — | **2.83%** | 20 fraudes/día con solo 200 alertas |
| **Threshold** | Global | Global | **Por canal** | Optimización quirúrgica |
| **Métricas** | Prec/Rec | Prec/Rec/F1 | + **Brier, ECE, Recall@k** | Visibilidad completa |
| **Artefacto** | .pkl simple | .pkl simple | .pkl con FE embebido | Autocontenido, no requiere código fuente |

### Mejora de señal: el salto conceptual

El cambio más importante entre v1 y v2 fue la **inyección de señal post-hoc**.
En v1 las features eran independientes de la etiqueta:

```python
# v1: probabilidad aditiva (señal débil)
prob = 0.10 + (0.05 if destino_alto_riesgo else 0) + (0.10 if cross_border else 0) + ...
# Cada feature aporta 2-25% → todo es ruido
```

En v2+: primero se etiqueta, luego se **modifican las features post-hoc**:

```python
# v2+: inyección de señal post-etiquetado
if is_fraud:
    row['dispositivo_reconocido'] = 0  # 60% de fraudes tienen dispositivo no reconocido
    row['cross_border'] = 1             # 55% de fraudes son cross-border
    row['numero_transacciones_ultima_hora'] = np.random.randint(6, 20)  # ráfagas
```

Esto crea **distribuciones genuinamente diferentes** entre fraude y legítimo,
permitiendo que el modelo aprenda patrones reales en lugar de correlaciones espurias.

Resultado: PR-AUC saltó de 0.35 a 0.96 — un **174% de mejora**.

---

## 4. Deep Dive Técnico

### 4.1 Feature Engineering — La pieza clave

El FeatureEngineer es un `sklearn.BaseEstimator, TransformerMixin` que implementa
`fit()` y `transform()`. Se entrena sobre train y se aplica sobre test/producción
sin data leakage.

**Estructura**: 44 columnas crudas → 69 features transformadas en 4 bloques:

#### Bloque 1 — Ratios financieros (8 features)

```python
# Ejemplos de ratios: capturan desviaciones del comportamiento normal
X['txn_vs_limit_pct'] = importe / limite_tarjeta          # ¿usa todo el límite?
X['outflow_inflow_ratio'] = volumen_saliente / volumen_entrante  # ¿drena la cuenta?
X['net_flow_30d'] = entrante - saliente                    # déficit financiero
X['balance_utilization'] = saldo_medio / (limite * 10)     # ¿vive al límite?
```

#### Bloque 2 — Flags geográficos (4 features)

```python
# Combinaciones de país + dispositivo = señales fortísimas
X['cross_border'] = (customer_country != operacion_pais).astype(int)
X['foreign_unknown_device'] = (cross_border == 1) & (dispositivo_reconocido == 0)
# ↑ 43.6% de fraudes vs 1% legítimas → segundo mayor discriminador del modelo
```

#### Bloque 3 — Features de sesión (6 features)

```python
# Detectan comportamiento automatizado o robótico
X['txn_por_minuto'] = num_tx_ultima_hora / (tiempo_ultima_tx / 60 + 1)
X['burst_rapido'] = (num_tx > 5) & (tiempo_ultima_tx < 300)  # ráfaga en 5min
X['alta_velocidad'] = (num_tx > 3) | (tiempo_ultima_tx < 30)
```

#### Bloque 4 — Desviación temporal con z-score real (3 features)

**Éste fue el cambio más importante de v3**. Antes teníamos `diff_importe_cliente`,
que comparaba el importe contra la media del cliente **ignorando la variabilidad**:

```python
# ❌ ANTES (v1/v2): ignoraba la desviación estándar
diff_importe_cliente = |importe - media_cliente| / media_cliente
# Un cliente con media 100€ y desviación 5€ → 200€ da diff=1.0 (anómalo)
# Un cliente con media 100€ y desviación 50€ → 200€ da diff=1.0 (normal, no es anómalo)
# ↑ El mismo ratio no distingue variabilidad normal de anomalía
```

**Ahora usamos un z-score real** usando `importe_medio_mensual` y `desviacion_estandar_mensual`:

```python
# ✅ AHORA (v3): z-score real
std = desviacion_estandar_mensual.replace(0, 1)  # evitar división por cero
diff_importe_zscore = |importe - importe_medio_mensual| / std
diff_importe_signed = (importe - importe_medio_mensual) / std  # con signo
importe_anomalo = (diff_importe_zscore > 3).astype(int)  # flag si |z| > 3
```

Esto tiene **3 ventajas fundamentales**:
1. **No necesita cold start**: usa campos del propio registro, no medias históricas del cliente
2. **Captura variabilidad natural**: un cliente variable no genera falsos positivos
3. **Interpretable**: z-score > 3 = anomalía real (regla estadística de 3 desviaciones)

#### Codificación de categóricas

Se usan dos técnicas en paralelo:

```python
# Frecuencia: counts normalizados
X[f'{col}_freq'] = X[col].map(freq_encoding).fillna(0)

# Target encoding: media de IS_FRAUD por categoría, con smooth
X[f'{col}_target'] = X[col].map(target_encoding).fillna(global_mean)
# ↑ La media global actúa como regularización para categorías no vistas
```

### 4.2 Pipeline de entrenamiento

El entrenamiento está en `scripts/regenerate_models.py`, que soporta v1, v2 y v3
con un solo código parametrizado por configuraciones:

```python
# Config de v3
{
    'name': 'v3',
    'data_path': 'data/dataset_fraude_v3.csv',  # 100K transacciones
    'savename': 'modelo_09_v3.pkl',
    'lgb_params': {
        'learning_rate': 0.1, 'min_child_samples': 100,
        'n_estimators': 200, 'num_leaves': 15, 'reg_lambda': 10,
    },
    'xgb_params': {
        'learning_rate': 0.05, 'max_depth': 3,
        'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 0,
    },
}
```

**Split y escalado**:

```python
# Train 60% / Val 20% / Test 20% con stratify
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# StandardScaler solo sobre numéricas (no sobre target encoding ni frecuencias)
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
X_val[num_feats] = scaler.transform(X_val[num_feats])

# KNNImputer con n=5 para rellenar nulos (pueden salir de divisiones por cero)
X_train[num_feats] = imputer.fit_transform(X_train[num_feats])
```

**El ensemble**: LightGBM + XGBoost con peso optimizado por validación:

```python
# Optimización del peso w en validación
for w in np.linspace(0, 1, 101):
    yprob_ens = w * yprob_lgb + (1 - w) * yprob_xgb
    prauc = average_precision_score(y_val, yprob_ens)
    if prauc > best_prauc:
        best_prauc, best_w = prauc, w

# v3 resultó: w=0.650 LGB + 0.350 XGB
```

**Threshold F2**: se optimiza sobre validación para maximizar F2-score,
que pondera recall 2× más que precisión:

```python
# F2-score = (5 * P * R) / (4 * P + R)
results_df['f2'] = (5 * precision * recall) / (4 * precision + recall + 1e-10)
best_f2_row = results_df.loc[results_df['f2'].idxmax()]
```

### 4.3 Recuperación ante desbalance extremo con scale_pos_weight

El desbalance en v3 es 3.5% fraude → ~96.5% legítimo.
Para compensarlo, usamos `scale_pos_weight`:

```python
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
# En v3: ~27.5 → el modelo penaliza 27.5× más los falsos negativos
```

### 4.4 Umbrales por canal — La innovación operativa de v3

Cada canal de pago tiene un perfil de riesgo distinto,
por lo que optimizamos un threshold diferente para cada uno:

```python
# Código de optimización en regenerate_models.py
canales = ['tarjeta', 'transferencia', 'bizum']
for canal in canales:
    mask = df_val['tipo_transaccion'].values == canal
    yc, pc = y_val[mask], yprob_val[mask]
    for t in np.linspace(0.01, 0.99, 200):
        f2 = (5 * P * R) / (4 * P + R)  # F2-score
        if f2 > best_f2_c:
            best_f2_c, best_t_c = f2, t
```

Resultados:

| Canal | Threshold | Efecto |
|---|---|---|
| tarjeta | 0.759 | Conservador: mucho volumen legítimo, pocos FP |
| transferencia | 0.724 | Agresivo: riesgo unitario alto, preferimos FN que dejar pasar |
| bizum | 0.783 | Intermedio: canal nuevo, datos limitados |

En producción, la API aplica el threshold correspondiente:

```python
thr = per_channel_thr.get(transaccion.tipo_transaccion, best_t)
is_fraud = int(prob >= thr)
```

### 4.5 Recall@k — La métrica que importa en producción

"No nos importa solo la precisión global. Importa cuántos fraudes capturamos
con los recursos limitados del equipo de analistas."

**¿Qué mide exactamente?** Recall@k ordena TODAS las transacciones por
probabilidad descendente y mira cuántos fraudes hay en el top k. Simula un
sistema de **triaje**: si el analista solo puede revisar k alertas y elige
las k más sospechosas del total, ¿cuántos fraudes pesca?

```python
# Simulación: 200.000 tx/día, 200 alertas/día → top 0.1%
k_pct = 0.001
k = max(1, int(len(y_test) * k_pct))
top_k_idx = np.argsort(yprob_test)[-k:]  # top k por probabilidad
frauds_in_top_k = int(y_test[top_k_idx].sum())
recall_at_k = frauds_in_top_k / total_frauds_test
```

**¿Y esto cómo se relaciona con lo que realmente ve el analista?**

En producción el flujo es:

```
Modelo puntúa 200.000 tx
       ↓
Threshold por canal → se flagran ~7.500 tx como fraude
       ↓
De esas 7.500, las 200 más sospechosas van al analista
       ↓
Analista revisa esas 200
```

El analista **no** recibe todas las 7.500, recibe las 200 con mayor probabilidad
del conjunto flagrado. Pero como cualquier transacción no flagrada tiene
probabilidad menor que cualquier flagrada, las 200 más probables del conjunto
flagrado **son exactamente las 200 más probables del total**. Por lo tanto,
el cálculo actual (top k del total) es equivalente al escenario real
(top k de las flagradas).

**Resultado v3**: Recall@k = 0.0283 — el modelo captura el 2.83% de todos
los fraudes en las primeras 200 alertas. **28× más que el azar** (0.1%).

**Pregunta típica**: "¿Solo 2.83%? Eso parece bajo."
**Respuesta**: "El recall@k no mide lo buena que es la detección, mide
cuánto fraude podemos cazar con **recursos limitados**. 200 alertas es el 0.1%
de las transacciones. Si pudiéramos revisar más, capturaríamos mucho más.
Mire la curva completa:"

| k (alertas/día) | % tx revisadas | Fraudes capturados | Recall@k | Precisión en alertas |
|---|---|---|---|---|
| 200 | 0.1% | 20 | 2.8% | 100% |
| 1.000 | 0.5% | 90 | 12.9% | 90% |
| 2.000 | 1.0% | 164 | 23.4% | 82% |
| 5.000 | 2.5% | 320 | 45.7% | 64% |
| 10.000 | 5.0% | 430 | 61.4% | 43% |

**Interpretación**: con 200 alertas/día la precisión es 100% pero solo
tocamos el 2.8% de los fraudes. Con 2.000 alertas/día ya capturamos el 23.4%
de todos los fraudes, aunque la precisión baja al 82% (18% son falsos positivos).
La limitación no es el modelo — es **cuántas alertas puede revisar el equipo**.

**¿Y el recall del 84.7% qué significa entonces?** Ese es el recall **global**
del modelo: de cada 100 fraudes, el modelo **detecta** ~85 (prob >= threshold).
Pero de esos 85, el analista solo alcanza a revisar una fracción. El resto
quedan en la cola de alertas sin revisar hasta que haya capacidad. El recall@k
mide cuántos de los 85 logramos **revisar** con los recursos disponibles."

### 4.6 Calibración — Brier Score y ECE

La probabilidad devuelta debe ser **calibrada**: una transacción con prob=0.8
debe ser fraudulenta el 80% de las veces.

```python
# Brier Score: MSE de probabilidad
brier = brier_score_loss(y_test, yprob_test)  # v3: 0.0252 (excelente)

# ECE: Expected Calibration Error en 10 bins
for i in range(n_bins):
    in_bin = (yprob_test >= bin_edges[i]) & (yprob_test < bin_edges[i+1])
    if in_bin.sum() > 0:
        acc = y_test[in_bin].mean()      # fraude observado
        conf = yprob_test[in_bin].mean() # probabilidad media predicha
        ece += (in_bin.sum() / len(y_test)) * abs(acc - conf)
# v3: ECE = 0.0616 (aceptable, mejorable con recalibración isotónica)
```

### 4.7 API — app.py v5.0.0

Arquitectura:

```python
sys.path.insert(0, 'scripts')  # Para pickle cargue FeatureEngineer
modelo = joblib.load('model/modelo_09_v3.pkl')

# Extraer componentes del artefacto
fe = modelo['fe']           # FeatureEngineer (entrenado)
scaler = modelo['scaler']   # StandardScaler
imputer = modelo['imputer'] # KNNImputer
lgb = modelo['lgb_model']   # LightGBM
xgb_m = modelo['xgb_model'] # XGBoost
best_w, best_t = modelo['best_w'], modelo['best_t']
per_channel_thr = modelo.get('per_channel_thresholds', {})
```

Pipeline de inferencia:

```python
def predecir(transaccion):
    df_row = pd.DataFrame([transaccion.model_dump()])

    # 1. Feature Engineering
    X = fe.transform(df_row)          # 44 → 69 features

    # 2. Extraer campos respuesta ANTES de escalar/imputar
    cross_border = int(X['cross_border'].values[0])
    txn_vs_limit_pct = float(X['txn_vs_limit_pct'].values[0])

    # 3. Escalar + Imputar
    X[num_feats] = scaler.transform(X[num_feats])
    X[num_feats] = imputer.transform(X[num_feats])

    # 4. Ensemble
    prob = best_w * lgb.predict_proba(X)[:, 1] + (1 - best_w) * xgb_m.predict_proba(X)[:, 1]

    # 5. Threshold por canal
    thr = per_channel_thr.get(transaccion.tipo_transaccion, best_t)
    is_fraud = int(prob >= thr)

    return Prediccion(...)  # 10 campos de salida
```

**Por qué extraemos campos calculados ANTES del escalado:**
`cross_border`, `txn_vs_limit_pct`, etc. son features calculadas por el FeatureEngineer
pero necesarias como output. Las extraemos antes de que el scaler las normalice.

### 4.8 Validación de entrada con Pydantic v2

Desde la versión 5.0.0, la API valida los datos de entrada con **`field_validator`**
y **constraints de Pydantic v2**, interceptando datos inválidos antes de que lleguen
al modelo:

```python
class Transaccion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, str_min_length=1)

    edad_cliente: int = Field(..., ge=0, le=120)
    importe_transaccion: float = Field(...)
    dispositivo_reconocido: int = Field(...)
    is_night: int = Field(...)
    is_weekend: int = Field(...)
    destino_alto_riesgo: int = Field(...)
    numero_pin_disponibles: int = Field(...)

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
```

**¿Qué hace cada cosa?**
- `ConfigDict(str_strip_whitespace=True, str_min_length=1)`: limpia espacios en blanco
  de todos los strings y rechaza strings vacíos (un `""` no tiene sentido como
  `id_transaccion` o `tipo_cliente`).
- `edad_cliente: int = Field(ge=0, le=120)`: Pydantic valida automáticamente que la
  edad esté entre 0 y 120. Si alguien envía `edad_cliente: 999`, la API devuelve
  HTTP 422 sin ejecutar el modelo.
- `field_validator` para `importe_transaccion`: rechaza importes ≤ 0.
- `field_validator` para campos binarios: garantiza que `dispositivo_reconocido`,
  `is_night`, `is_weekend` y `destino_alto_riesgo` sean exactamente 0 o 1.
- `field_validator` para `numero_pin_disponibles`: rechaza valores negativos.

**Valor para el sistema:**
1. **Cero coste de inferencia en datos inválidos**: el modelo nunca procesa
   transacciones con campos imposibles (edad 999, importe negativo, flag=3).
2. **Errores claros y automáticos**: el HTTP 422 de FastAPI incluye el campo
   y el motivo exacto del rechazo, sin try/catch manual.
3. **Documentación viva**: los constraints aparecen en el schema OpenAPI
   (`/docs`), visibles para Full Stack y Ciberseguridad.
4. **Sin riesgo de ruptura**: todos los validadores son conservadores
   (solo rechazan lo que seguro es error). Ningún cambio en la lógica
   de negocio existente.

También se añadió validación en la respuesta:

```python
class Prediccion(BaseModel):
    prob_fraud: float

    @field_validator('prob_fraud')
    @classmethod
    def prob_entre_0_y_1(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('prob_fraud debe estar entre 0.0 y 1.0')
        return v
```

Garantiza que nunca se devuelva una probabilidad fuera de rango aunque
haya un error numérico en el ensemble.

### 4.10 Generación de datos de prueba con perfiles de riesgo — `generar_muestra_sin_etiqueta.py`

El script genera transacciones sintéticas **sin etiqueta** (sin `IS_FRAUD`) para
probar la API o el pipeline de inferencia sin depender de datos reales.
Incluye **3 perfiles de riesgo** para evaluar cómo responde el modelo ante
distintos patrones:

```python
PERFILES = {
    'mixto': {
        'label': 'Muestra mixta (patrón estándar)',
        'p_dispositivo_reconocido': 0.85,
        'p_cross_border': 0.10,
        'p_destino_alto_riesgo': 0.12,
        'p_estados_tarjeta': [0.80, 0.05, 0.08, 0.04, 0.03],
        'importe_log_mean': 5,      # ~150€
        'poisson_tx_hora': 2,
        'p_night': 0.15,
    },
    'sospechoso': {
        'p_dispositivo_reconocido': 0.50,
        'p_cross_border': 0.40,
        'importe_log_mean': 6,      # ~400€
        'poisson_tx_hora': 5,
    },
    'fraude': {
        'p_dispositivo_reconocido': 0.20,
        'p_cross_border': 0.70,
        'importe_log_mean': 7,      # ~1100€
        'poisson_tx_hora': 10,
    },
}
```

Cada perfil modifica las distribuciones de generación para crear señales
de riesgo más o menos marcadas. Esto permite **testear el comportamiento
del modelo en diferentes escenarios** antes de desplegar.

```powershell
# Un perfil específico
python scripts/generar_muestra_sin_etiqueta.py --perfil fraude
# → data/muestra_fraude.csv + .json

# Todos a la vez
python scripts/generar_muestra_sin_etiqueta.py --perfil todo
# → muestra_mixto, muestra_sospechoso, muestra_fraude (CSV+JSON cada uno)

# Evaluación rápida contra la API
python -c "
import json, requests
for p in ['mixto', 'sospechoso', 'fraude']:
    with open(f'data/muestra_{p}.json') as f:
        r = requests.post('http://localhost:8000/predict/batch', json=json.load(f))
    d = r.json()
    print(f'{p}: {d[\"fraudes\"]}/{d[\"total\"]} fraudes ({d[\"fraudes\"]/d[\"total\"]*100:.1f}%)')
"
```

### 4.11 Simulación de producción con `evaluacion_rondas.py`

El script `evaluacion_rondas.py` es un **simulador de producción** que:

1. Genera transacciones en lotes de 100 (simula la «ronda» semanal de Ciberseguridad)
2. Ejecuta el pipeline completo (FE → scaler → imputer → ensemble → threshold)
3. Acumula métricas por ronda
4. Permite inyectar **drift** (suave, abrupto, de concepto)
5. Detecta drift con **KS-test** sobre las predicciones

```python
# Escenarios:
# --drift suave:   cambio gradual en features de fraude (ronda 20-40)
# --drift abrupto: cambio brusco en ronda 30
# --drift concepto: la relación features → fraude cambia (nuevos patrones)

# KS-test para detectar drift
stat, p_value = ks_2samp(probs_round, probs_historico)
if p_value < 0.05:
    print(f"⚠️ Drift detectado en ronda {ronda} (p={p_value:.4f})")
```

### 4.9 Dockerfile — Despliegue autocontenido

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY scripts ./scripts        # FeatureEngineer disponible en /app/scripts/
COPY model/modelo_09_v3.pkl ./model/
COPY app.py .
RUN pip install fastapi uvicorn joblib pandas scikit-learn lightgbm xgboost
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Clave**: `COPY scripts ./scripts` antes de `COPY app.py .` para que
`sys.path.insert(0, 'scripts')` en app.py funcione y el pickle pueda
deserializar la clase `FeatureEngineer`.

---

## 5. Métricas y Resultados

### Comparativa v2 vs v3 (datos de test)

| Métrica | v2 (15% fraude) | v3 (3.5% fraude) | Interpretación |
|---|---|---|---|
| **PR-AUC** | 0.9622 | 0.9013 | Excelente: mantener 0.90 con 1/5 de fraude es muy bueno |
| **AUC-ROC** | 0.9864 | 0.9869 | Idéntico: el ranking no se degrada |
| **Precisión** | 77.0% | **78.7%** | Mejor: menos falsos positivos |
| **Recall** | 94.7% | 84.7% | Baja porque el threshold sube (de 0.314 a ~0.75) |
| **F1** | 0.8495 | 0.8161 | Compromiso razonable |
| **Brier** | 0.0267 | **0.0252** | Mejor calibración: las probabilidades son más fiables |
| **ECE** | 0.0319 | 0.0616 | Empeora: hay que recalibrar con isotónico |
| **Recall@k** | — | 0.0283 | 20 fraudes/día con 200 alertas, 100% precisión en alertas |
| **Threshold** | 0.314 | 0.75 (global) | Más conservador porque hay menos fraude |
| **Peso LGB** | 0.670 | 0.650 | Estable: el ensemble converge a ~2/3 LGB + 1/3 XGB |

### Interpretación de negocio

"Si nuestra precisión es 78.7%, significa que de cada 100 transacciones que
nuestro sistema marca como fraude, ~79 lo son realmente. El analista revisa
21 falsos positivos por cada 100 alertas, lo cual es operativamente asumible.

El recall de 84.7% significa que detectamos ~85 de cada 100 fraudes reales.
Los ~15 que se escapan son los que más se parecen al comportamiento legítimo
(importes normales, dispositivo reconocido, horario diurno).

Con Recall@k, el valor más tangible: de los ~7.000 fraudes semanales,
el analista con 1.000 alertas/semana captura ~140 fraudes (2.83% de recall@k
× 7000 × 5 días / 0.1% ratio de alerta). Pero esas 1.000 alertas tienen
precisión 100% —ninguna es pérdida de tiempo."

### Gráficos recomendados para la presentación

1. **Curva PR-AUC** comparando v1 vs v2 vs v3
2. **Gráfico de calibración** (confianza vs frecuencia observada) para v3
3. **Feature importance** top 15 (SHAP o weight-based)
4. **Recall@k** vs % de alertas (curva que muestra cómo crece la captura)
5. **Distribución de probabilidades** separando fraude vs legítimo

---

## 6. Arquitectura de Producción

### Componentes

```
┌──────────┐    POST /predict    ┌──────────┐
│  Cliente  │ ──────────────────→ │   API     │
│  (Web)    │ ←────────────────── │ FastAPI   │
└──────────┘    {10 campos}       └────┬─────┘
                                        │
                            ┌───────────┴───────────┐
                            │   Modelo v3 (.pkl)     │
                            │   FeatureEngineer      │
                            │   StandardScaler       │
                            │   KNNImputer           │
                            │   LightGBM + XGBoost   │
                            └───────────────────────┘
                                        │
                            ┌───────────┴───────────┐
                            │   PostgreSQL           │
                            │   (histórico +         │
                            │    target_final para   │
                            │    reentrenamiento)    │
                            └───────────────────────┘
```

### Pipeline de reentrenamiento

```
Ciberseguridad genera
nuevas rondas de fraude
        ↓
Data Science etiqueta
con target_final real
        ↓
Ejecuta regenerate_models.py
con nuevo dataset (v4, v5...)
        ↓
Se genera nuevo .pkl
        ↓
Se despliega en producción
(swap de archivo + reload API)
```

### Estrategia de versionado

| Modelo | Archivo | Dataset | Features | Threshold |
|---|---|---|---|---|
| v1 | modelo_07_v1.pkl | dataset_fraude.csv (10K, 15%) | 67 | F2 global |
| v2 | modelo_08_v2.pkl | dataset_fraude_v2.csv (10K, 15%) | 67 | F2 global |
| v3 | modelo_09_v3.pkl | dataset_fraude_v3.csv (100K, 3.5%) | 69 | F2 por canal |

Cada .pkl es autocontenido: incluye el código fuente de FeatureEngineer embebido,
el scaler, el imputer, los modelos, los pesos del ensemble, y los thresholds.

---

## 7. Preguntas Frecuentes y Respuestas

### 7.1 Negocio

**P: ¿Por qué solo 3% de fraude en v3 si antes era 15%? ¿No es menos exigente?**

R: El 15% era artificial —venía del generador original que ponía una tasa
fija para tener suficientes casos. En la realidad, el fraude bancario ronda
el 1-5% (dependiendo del canal y país). Modelar con 15% infla artificialmente
el PR-AUC y da una falsa sensación de rendimiento. Con 3% nos acercamos más
a la realidad y el modelo aprende a ser más selectivo. De hecho, la precisión
mejoró (77% → 78.7%) porque el modelo tiene que esforzarse más para identificar
fraudes reales en lugar de simplemente disparar a todo.

**P: ¿Cuánto cuesta operar este sistema?**

R: En recursos: el modelo completo cabe en ~200MB RAM, la inferencia es
~50ms por transacción en CPU (sin GPU). Con 200.000 tx/día, necesitamos
~3 horas CPU/día, asumible por cualquier instancia estándar de cloud.
El coste principal es el analista revisando alertas —por eso priorizamos
precisión y recall@k.

**P: ¿Qué pasa si el fraude cambia de patrón?**

R: Tenemos dos líneas de defensa:
1. **Detección de drift**: `evaluacion_rondas.py` ejecuta KS-test entre rondas
   y alerta si la distribución de probabilidades cambia significativamente.
2. **Reentrenamiento continuo**: Cada ronda de Ciberseguridad genera nuevos
   patrones de fraude que se incorporan al dataset. Cuando el drift es
   significativo, se regenera el modelo con los nuevos datos.

**P: ¿Por qué umbrales diferentes por canal?**

R: Cada canal tiene un perfil de riesgo distinto:
- **Tarjeta**: mucho volumen, muchos pagos legítimos pequeños → threshold alto
  (0.759) para no bloquear compras normales.
- **Transferencia**: montos más altos, menos frecuencia → threshold más bajo
  (0.724) porque el coste de dejar pasar un fraude es mayor.
- **Bizum**: canal nuevo con pocos datos históricos → threshold intermedio
  (0.783) como posición conservadora hasta tener más información.

**P: ¿Qué pasa si no tenemos datos de un canal (ej. bizum recién lanzado)?**

R: La API tiene fallback al threshold global: si `per_channel_thr.get(canal)`
devuelve None, usa `best_t` (0.75). A medida que el canal acumula datos,
se regenera el modelo con ese canal incluido y se obtiene su threshold específico.

**P: ¿Cómo se explica una decisión de fraude a un cliente?**

R: El output no es solo binario. Devolvemos 10 campos con contexto:
`es_transfronteriza`, `ratio_imp_limite`, `intensidad_tx`, `flujo_neto_30d`.
Si un cliente reclama, el analista puede ver: "la transacción era cross-border
desde un dispositivo no reconocido, con un importe 4× superior a tu media,
y se hicieron 12 transacciones en la última hora." Eso permite una explicación
concreta y humana, no un "el algoritmo dijo que no".

### 7.2 Técnicas

**P: ¿Por qué LightGBM + XGBoost y no un solo modelo?**

R: Los dos modelos tienen fortalezas complementarias:
- **LightGBM**: mejor con outliers y datos ruidosos, más rápido de entrenar,
  maneja bien categorías con alta cardinalidad. Peso en ensemble: 65%.
- **XGBoost**: mejor calibración natural, maneja mejor datos con skew extremo
  (nuestra tasa 3.5%). Peso: 35%.

Combinarlos da mejor PR-AUC que cualquiera por separado (~+0.02-0.03).
Además, reduce la varianza: si un modelo falla en una región del espacio,
el otro lo compensa.

**P: ¿Por qué StandardScaler y no RobustScaler? Hay outliers en los datos.**

R: Lo probamos. RobustScaler (que usa mediana e IQR) no mejoró los resultados
porque:
1. Las features más importantes para el modelo (cross_border, foreign_unknown_device)
   son binarias, no les afecta el escalado.
2. Las features numéricas con outliers (importe_transaccion, volumenes) pasan
   por log-transform antes de escalar, lo que reduce el impacto de outliers.
3. KNNImputer funciona mejor con datos normalizados por StandardScaler porque
   las distancias euclídeas son más estables.

**P: ¿Por qué KNNImputer y no SimpleImputer (media/mediana)?**

R: KNNImputer con n=5 estima valores perdidos basándose en los 5 vecinos más
cercanos, lo que preserva correlaciones entre features. Por ejemplo, si falta
`saldo_actual` pero tenemos `saldo_medio_30_dias` y `volumen_entrante`, el
KNNImputer usará esas correlaciones para imputar mejor que simplemente poner
la media. En nuestro caso, los nulos suelen venir de divisiones por cero en
`_safe_ratio`, y KNN da mejor imputación que la media.

**P: ¿Cómo evitáis el data leakage en feature engineering?**

R: El `FeatureEngineer` implementa `fit()` y `transform()` como cualquier
transformer de sklearn. En `fit()` se calculan:
- Frecuencias de categóricas (sobre train)
- Target encoding (sobre train)
- Medias por cliente para features temporales (sobre train)

En `transform()` se **aplican** esos valores sin recalcular nada.
Además, el target encoding solo se calcula si hay target disponible
(entrenamiento) y usa la media global como regularización para categorías
nuevas en test/producción.

**P: ¿Por qué el ECE subió de 0.03 a 0.06 en v3?**

R: Porque hay menos datos de fraude en v3 (3.5% vs 15%). Con menos ejemplos
positivos, la estimación de probabilidad en cada bin tiene más varianza.
Además, el threshold óptimo subió de 0.31 a 0.75, lo que significa que
el modelo concentra las probabilidades en un rango más estrecho (0.7-0.9),
dejando bins con pocas muestras. Solución: recalibración isotónica en el
próximo ciclo.

**P: ¿Cómo se calcula el Recall@k exactamente y por qué es tan bajo?**

R: Recall@k ordena todas las transacciones del test por probabilidad descendente,
toma el top k (k = test_size × 0.001, simulando el 0.1% de alertas diarias),
y calcula qué fracción de todos los fraudes del test están en ese top k.

```
k = 0.1% × 20,000 test ≈ 20 alertas
Fraudes totales en test ≈ 700 (3.5% de 20,000)
Fraudes en top 20 ≈ 20
Recall@k = 20 / 700 = 0.0283
```

**Parece bajo** porque 700 fraudes y solo 20 en las primeras 20 alertas.
Pero 20 es 28× más que el azar (0.7 fraudes esperados si fuera aleatorio).
Además, esas 20 tienen precisión 100%. Las 680 restantes no se pierden:
el threshold por canal captura ~84.7% de todos los fraudes (recall global),
y las alertas se escalonan: las 200 primeras con precisión 100%, las siguientes
con precisión decreciente hasta la línea de threshold.

**P: ¿Por qué eliminaste `high_ratio_redondeado`?**

R: Porque aparecía en solo el 0.3% de fraudes del dataset, y la condición
`importe % 100 == 0` destruía la señal del ratio alto. Una transacción de
1950€ con límite 2000€ tiene ratio 0.975 (alta), pero 1950 % 100 = 50 ≠ 0,
con lo que `high_ratio_redondeado` era 0. La feature `txn_vs_limit_pct`
ya capturaba toda la señal sin esta restricción artificial.

**P: ¿Cómo se generan los datos sintéticos? ¿Son realistas?**

R: Usamos Faker con una estructura jerárquica de 4 niveles:
cliente → cuenta → tarjeta → transacción. Cada nivel tiene sus propias
distribuciones y correlaciones. Por ejemplo:
- Clientes `premium` tienen límites de tarjeta más altos y más volumen mensual
- La edad del cliente correlaciona con el tipo de autenticación (mayores usan más firma)
- El país del cliente determina moneda, regiones y probabilidad de operaciones internacionales

Además, el proceso de **inyección de señal post-hoc** (feature engineering inverso)
crea distribuciones realistas donde los fraudes tienen:
- 55% cross-border vs 8% legítimos
- 60% dispositivo no reconocido vs 15% legítimos
- 35% tarjeta robada/extraviada vs 7% legítimos

Esto genera patrones que un modelo real puede aprender, muy distinto de la
probabilidad aditiva plana del v1.

**P: ¿Qué métrica usáis para decidir si un nuevo modelo es mejor que el anterior?**

R: No miramos una sola métrica. Usamos una **matriz de decisión**:

1. **PR-AUC**: la principal. Si baja más de 0.02 con mismo dataset, rechazamos.
2. **Recall@k**: si mejora, es candidato fuerte.
3. **Brier + ECE**: la calibración debe ser mejor o igual.
4. **Precisión en producción**: simulamos con datos sin etiqueta y revisamos
   las alertas manualmente antes de desplegar.

En v3, PR-AUC bajó de 0.96 a 0.90 porque la tasa de fraude cambió de 15% a 3.5%.
Pero la precisión mejoró y recall@k dio visibilidad operativa que antes no teníamos.
El criterio fue: "modelo más realista con mejores métricas operativas (recall@k,
calibración) aunque PR-AUC sea menor por el desbalance más extremo."

**P: ¿Habéis considerado deep learning? (RNN, Transformer)**

R: Lo evaluamos para v2 y decidimos no usarlo por:
1. **Volumen de datos**: 10K-100K registros es poco para DL. Los gradient-boosted
   trees funcionan mejor con datasets pequeños/medianos.
2. **Interpretabilidad**: con SHAP values podemos explicar cada decisión.
   Con una red neuronal sería mucho más opaco.
3. **Latencia**: una red neuronal añade latencia de inferencia sin ganancia
   clara de rendimiento para este problema.
4. **Mantenimiento**: gradient boosting requiere menos hyperparameter tuning
   y es más robusto ante cambios en los datos.

Dicho esto, si el volumen creciera a millones de transacciones y hubiera
secuencias temporales largas, un Transformer (o TabNet) podría ser relevante.
Para el problema actual, XGBoost + LightGBM es el estado del arte en datos tabulares.

**P: ¿Qué pasa si mañana aparece un canal nuevo (ej. cripto)?**

R: El sistema maneja canales no vistos de dos formas:
1. **A nivel de threshold**: la API usa el threshold global (fallback) si no
   hay threshold específico para ese canal.
2. **A nivel de feature engineering**: `tipo_transaccion` se codifica con
   target encoding y frequency encoding. Si aparece un valor nuevo (ej. "cripto"),
   el frequency encoding le asigna 0 y el target encoding le asigna la media
   global (≈3.5% fraude). El modelo tratará ese canal como "desconocido",
   que suele ser señal de riesgo en sí mismo.

A medio plazo, cuando el canal acumule datos, se reentrena el modelo y se
incluye en la optimización de thresholds por canal.

**P: ¿Cómo se mantiene el modelo en producción? ¿Hay monitoreo?**

R: Tenemos varios niveles:
1. **Monitoreo de input**: distribución de features crudas (media, std, null rate)
   por ventana de tiempo. Si `importe_transaccion` salta de media 150€ a 2000€,
   algo cambió.
2. **Monitoreo de output**: distribución de `prob_fraud`, ratio de `is_fraud=1`,
   y PSI (Population Stability Index) entre ventanas.
3. **Drift detection**: KS-test sobre las predicciones de cada ronda vs el histórico.
4. **Alertas**: si PSI > 0.1 o KS-test p < 0.05, se dispara una alerta para
   revisar y potencialmente reentrenar.

El script `evaluacion_rondas.py` implementa exactamente este pipeline de
monitoreo offline para simular producción antes de desplegar.

**P: ¿Cómo integráis el feedback de los analistas?**

R: Cuando un analista confirma o rechaza una alerta, ese `target_final` se
guarda en la base de datos. En el próximo ciclo de reentrenamiento:
1. Se leen las transacciones con `target_final` (reales, no sintéticas)
2. Se mezclan con los datos sintéticos de Ciberseguridad
3. Se reentrena el modelo con esos datos reales
4. El modelo "aprende" de los patrones de fraude real confirmados por humanos

Esto cierra el círculo: datos sintéticos para arrancar → feedback humano
para refinar → modelo más preciso → menos falsos positivos → analistas más
eficientes → más feedback de calidad.

**P: ¿El modelo 09_v3 es el definitivo?**

R: No. La v3 es nuestra mejor versión hasta ahora, pero el fraude evoluciona.
El plan es:
1. **v3.1**: Recalibración isotónica para bajar ECE < 0.05
2. **v4**: Incorporar rondas reales de Ciberseguridad, SHAP analysis para
   feature selection, y thresholds dinámicos que se ajusten con el volumen
   diario de transacciones
3. **v5**: Si los datos reales lo justifican, explorar modelos secuenciales
   o incluir features de red social entre cuentas (pagar a una cuenta que
   acaba de recibir fraude de otra)

**P: ¿Cuál fue el mayor desafío técnico del proyecto?**

R: Sin duda la **generación de datos sintéticos realistas**. Al principio (v1)
los datos eran demasiado limpios: las features eran independientes entre sí
y de la etiqueta. El modelo no aprendía nada (PR-AUC 0.35).

Tuvimos que invertir el proceso: primero etiquetar (con una tasa realista),
y luego **deformar las features post-hoc** para que reflejaran patrones de
fraude reales. Esto es conceptualmente más difícil que simplemente ajustar
una fórmula de probabilidad, porque hay que entender qué correlaciones existen
en el mundo real —por ejemplo, que el fraude cross-border suele venir de
dispositivos no reconocidos, o que las ráfagas de transacciones en minutos
son típicas de bots.

Una vez resuelto eso, el resto del pipeline fue estándar: feature engineering,
ensemble, threshold optimization. La clave estaba en los datos.

---

## Apéndice: Arquitectura del artefacto .pkl

El modelo guardado es un **diccionario Python** con todo lo necesario
para inferencia sin depender del código fuente:

```python
artifact = {
    '_fe_source': FE_SOURCE,       # Código fuente de FeatureEngineer (string)
    'fe': fe,                      # FeatureEngineer entrenado (objeto)
    'scaler': scaler,              # StandardScaler entrenado
    'imputer': imputer,            # KNNImputer entrenado
    'lgb_model': lgb_model,        # LightGBM entrenado
    'xgb_model': xgb_model,        # XGBoost entrenado
    'best_w': best_w,              # Peso del ensemble (LGB)
    'best_t': best_t,              # Threshold global
    'num_feats': num_feats,        # Lista de features numéricas
    'per_channel_thresholds': {...},  # Thresholds por canal
    'metadata': {                  # Métricas de evaluación
        'pr_auc': 0.9013,
        'roc_auc': 0.9869,
        'precision': 0.787,
        'recall': 0.847,
        'brier': 0.0252,
        'ece': 0.0616,
        'recall_at_k': 0.0283,
    },
    'model_version': 'v3',         # Para trazabilidad
}
```

**Por qué es importante `_fe_source`**: cuando joblib serializa el
FeatureEngineer, necesita la clase disponible en el `sys.path` al cargar.
Si por algún motivo el código fuente no está disponible (e.g., en un
entorno serverless o un contenedor sin `scripts/`), joblib puede usar
`_fe_source` para reconstruir la clase mediante `exec()`.

---

## Apéndice 2: Cómo hacer una demo en vivo

### Demo 1: API funcionando

```powershell
# 1. Arrancar la API
uvicorn app:app --host 0.0.0.0 --port 8000

# 2. Health check
curl http://localhost:8000/health
# → {"status":"ok","version":"5.0.0","modelo":"modelo_09_v3.pkl",
#    "threshold_global":0.75,"thresholds_canal":{"tarjeta":0.759,...}}

# 3. Predecir una transacción
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "id_transaccion": "demo-001",
    "id_cliente": "cliente-001",
    "tipo_cliente": "persona",
    "edad_cliente": 35,
    "customer_country": "ES",
    "customer_region": "Centro",
    "tenure": 365,
    "importe_medio_mensual": 500,
    "desviacion_estandar_mensual": 150,
    "media_transacciones_al_dia": 3,
    "numero_fraudes_ultimo_ano": 0,
    "id_cuenta": "cuenta-001",
    "cuenta_origen": "ES123456789",
    "estado_cuenta": "activa",
    "saldo_actual": 2500,
    "saldo_medio_30_dias": 2200,
    "volumen_entrante_30_dias": 3000,
    "volumen_saliente_30_dias": 2800,
    "numero_transferencias_recibidas_7_dias": 3,
    "numero_transferencias_enviadas_7_dias": 2,
    "id_tarjeta": "tarjeta-001",
    "estado_tarjeta": "activa",
    "fecha_creacion_tarjeta": "2023-01-15",
    "antiguedad_tarjeta_dias": 365,
    "limite_importe_transacciones": 2000,
    "veces_superar_limite_7_dias": 0,
    "tipo_transaccion": "transferencia",
    "fecha_hora": "2026-05-27T14:30:00",
    "is_night": 0,
    "is_weekend": 0,
    "tiempo_desde_ultima_transaccion": 3600,
    "numero_transacciones_ultima_hora": 1,
    "importe_transaccion": 150,
    "metodo_autenticacion": "PIN",
    "numero_pin_disponibles": 3,
    "identificador_dispositivo_fingerprint": "device-001",
    "dispositivo_reconocido": 1,
    "operacion_pais": "ES",
    "operacion_region": "Centro",
    "direccion_ip_origen": "86.34.12.179",
    "geolocalizacion": "40.4168,-3.7038",
    "cuenta_destino": "ES987654321",
    "destino_alto_riesgo": 0
  }'
```

### Demo 2: Batch inference

```powershell
python scripts/prediccion_lote.py `
  --input data/dataset_fraude_v3.csv `
  --output data/predicciones_v3.csv `
  --modelo v3
```

### Demo 3: Evaluación por rondas

```powershell
python scripts/evaluacion_rondas.py `
  --modelo v3 --rondas 50 --drift suave --output resultados.csv
```

### Demo 4: Generar muestra sin etiqueta (múltiples perfiles)

```powershell
# Muestra estándar (mixto, 200 tx)
python scripts/generar_muestra_sin_etiqueta.py
# → data/muestra_sin_etiqueta.csv + data/muestra_sin_etiqueta.json

# Perfil fraudulento (señales fuertes de fraude)
python scripts/generar_muestra_sin_etiqueta.py --perfil fraude
# → data/muestra_fraude.csv + data/muestra_fraude.json

# Perfil sospechoso (señales intermedias)
python scripts/generar_muestra_sin_etiqueta.py --perfil sospechoso
# → data/muestra_sospechoso.csv + data/muestra_sospechoso.json

# Generar todos los perfiles a la vez
python scripts/generar_muestra_sin_etiqueta.py --perfil todo
# → Crea 3 pares CSV+JSON: muestra_mixto, muestra_sospechoso, muestra_fraude
```

**Perfiles disponibles:**

| Perfil | dispositivo_reconocido | cross_border | importe medio | tx/hora |
|---|---|---|---|---|
| `mixto` | 85% | 10% | ~150€ | 2 |
| `sospechoso` | 50% | 40% | ~400€ | 5 |
| `fraude` | 20% | 70% | ~1100€ | 10 |

### Demo 5: Pasar JSON al modelo para evaluar

#### Opción A — vía API (una transacción)

```powershell
# Con curl (PowerShell)
$body = Get-Content data/muestra_sin_etiqueta.json -Raw | ConvertFrom-Json
$primera = $body[0] | ConvertTo-Json
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d $primera
```

#### Opción B — vía API (lote, todas las transacciones)

```powershell
curl -X POST http://localhost:8000/predict/batch `
  -H "Content-Type: application/json" `
  -d (Get-Content data/muestra_sin_etiqueta.json -Raw)
```

#### Opción C — vía batch script (CSV)

```powershell
python scripts/prediccion_lote.py `
  --input data/muestra_sin_etiqueta.csv `
  --output data/resultados_muestra.csv `
  --modelo v3
```

#### Opción D — Script Python rápido para evaluar los 3 perfiles

```python
# evaluar_perfiles.py — comparar cómo responde el modelo a cada perfil
import json, requests

API = "http://localhost:8000/predict/batch"

for perfil in ['mixto', 'sospechoso', 'fraude']:
    with open(f'data/muestra_{perfil}.json', encoding='utf-8') as f:
        data = json.load(f)

    resp = requests.post(API, json=data).json()
    fraudes = resp['fraudes']
    total = resp['total']
    print(f"{perfil:12s}: {fraudes}/{total} marcadas como fraude ({fraudes/total*100:.1f}%)")
```

Salida esperada:
```
mixto       : 0/200 marcadas como fraude (0.0%)
sospechoso  : 8/200 marcadas como fraude (4.0%)
fraude      : 67/200 marcadas como fraude (33.5%)
```

---

## Apéndice 3: Checklist de revisión técnica

Antes de desplegar un nuevo modelo, verificar:

- [ ] ¿FeatureEngineer tiene `fit()` y `transform()`? Sin data leakage.
- [ ] ¿El .pkl incluye `_fe_source`? Por si el código fuente no está disponible.
- [ ] ¿Los thresholds por canal se optimizan sobre validación, no sobre test?
- [ ] ¿`sys.path.insert(0, 'scripts')` está antes de `joblib.load` en todos los scripts?
- [ ] ¿El dockerfile copia `scripts/` antes de copiar `app.py`?
- [ ] ¿Las rutas en el dockerfile son relativas al WORKDIR (`/app`)?
- [ ] ¿`num_feats` incluye solo columnas numéricas (no target encoding ni freq)?
- [ ] ¿El scaler se fittea sobre train y solo transforma en producción?
- [ ] ¿El imputer usa KNN con n_neighbors razonable (5)?
- [ ] ¿La API extrae campos de respuesta (cross_border, etc.) ANTES de escalar?
- [ ] ¿El health endpoint reporta versión, modelo, thresholds, recall@k?
- [ ] ¿`tipo_transaccion` en la request permite elegir threshold por canal?
- [ ] ¿Hay fallback al threshold global si el canal no tiene threshold específico?
- [ ] ¿El Brier Score es < 0.10? (v3: 0.0252 ✅)
- [ ] ¿El ECE es < 0.10? (v3: 0.0616 ⚠️, objetivo < 0.05)
- [ ] ¿El Recall@k está documentado para que negocio entienda el impacto operativo?
- [ ] ¿Los datos sintéticos tienen 44 columnas de salida (compatibles con v1/v2)?
- [ ] ¿`night_velocity` es int (no bool) para que entre en `num_feats`?
- [ ] ¿`IS_FRAUD` se elimina del df ANTES de FeatureEngineer en entrenamiento?
