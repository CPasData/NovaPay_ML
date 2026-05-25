# Análisis de Residuos y Evaluación de Modelos de Fraude

## 1. ¿Qué son los residuos en clasificación binaria?

En regresión lineal, el residuo es `y_true - y_pred`. En clasificación binaria hay múltiples
definiciones útiles, cada una con un propósito distinto:

| Tipo | Fórmula | Rango | Propósito |
|------|---------|-------|-----------|
| **Residuo crudo** | `y_true - y_pred` | {-1, 0, +1} | −1 = FP, 0 = acierto, +1 = FN |
| **Residuo de probabilidad** | `y_true - y_prob` | [-1, +1] | Magnitud del error de probabilidad |
| **Residuo de Pearson** | `(y_true - y_prob) / sqrt(y_prob*(1-y_prob))` | (-∞, +∞) | Error estandarizado por varianza esperada |
| **Residuo de deviance** | sign(y_true-y_prob) · sqrt(-2 · log-likelihood) | (-∞, +∞) | Contribución a la pérdida total |

---

## 2. Análisis de residuos por segmentos

**Objetivo**: detectar segmentos donde el modelo falla sistemáticamente.

Para cada segmento categórico (`tipo_transaccion`, `customer_country`, `metodo_autenticacion`,
`operacion_pais`, `tipo_cliente`, `rango_importe`, `franja_horaria`, etc.):
- Tasa de FP, FN, precision, recall
- Distribución de residuos de Pearson
- PR-AUC y AUC-ROC dentro del segmento

**Forma de detectar**: calcular precisión y recall por segmento; si un segmento tiene
recall significativamente menor que la media, el modelo tiene un sesgo hacia ese segmento.

**Aplicación en fraude**: si las transferencias tienen recall 60% vs. 90% en tarjeta,
el modelo penaliza sistemáticamente ese tipo de transacción.

---

## 3. Análisis temporal y detección de drift

**Objetivo**: detectar si el modelo se degrada con el tiempo (concept drift o data drift).

### 3.1 Drift en scores
Calcular PSI (Population Stability Index) entre la distribución de `y_prob` del lote actual
vs. el lote de entrenamiento. PSI > 0.1 indica drift significativo.

### 3.2 Drift en residuos
- Media móvil del residuo de probabilidad por ronda
- KS-test entre distribución de scores actual vs. baseline
- Si los residuos se vuelven sistemáticamente positivos (más FN) o negativos (más FP),
  el modelo está desactualizado.

### 3.3 Drift en features
Para cada feature numérica: KS-test entre lote actual y entrenamiento.
Para categóricas: chi-cuadrado.

---

## 4. Análisis de errores por feature

**Objetivo**: identificar qué features correlacionan con errores.

### 4.1 Binning por feature
Para cada feature numérica:
1. Dividir en deciles
2. Calcular tasa de error (FP + FN) por decil
3. Identificar deciles con error > 2× la media global

### 4.2 Partial dependence plots de residuos
Graficar `E[residuo | feature]` — muestra cómo cambia el error a lo largo del rango
de una feature. Si la curva se desvía sistemáticamente de cero, hay sesgo.

### 4.3 Interaction analysis
Buscar pares de features donde el error conjunto es mayor que la suma de errores
individuales (sinergia de error). Útil para encontrar patrones complejos que el modelo
no captura.

---

## 5. Análisis de calibración

**Objetivo**: verificar que `y_prob` ≈ frecuencia real de fraude.

### 5.1 Calibration plot (reliability diagram)
1. Dividir `y_prob` en 10-20 bins
2. Para cada bin: frecuencia observada de fraude
3. Ideal: puntos en la diagonal
4. Desviaciones sistemáticas → recalibrar

### 5.2 Brier score
`Brier = mean((y_true - y_prob)^2)` — mide calibración + discriminación.
Brier < 0.1 es aceptable. Se puede descomponer en:
- **Refinement**: pérdida por falta de resolución
- **Calibration**: pérdida por mala calibración
- **Uncertainty**: varianza intrínseca del target

### 5.3 Expected Calibration Error (ECE)
Diferencia media absoluta entre probabilidad predicha y frecuencia observada,
ponderada por el número de muestras en cada bin.

---

## 6. Técnicas específicas para fraude

### 6.1 Matriz de confusión por segmento de riesgo
Dividir transacciones en segmentos por `y_prob` (bajo <0.1, medio 0.1-0.5, alto >0.5)
y calcular TP/FP/TN/FN dentro de cada segmento. Permite ver si los FP se concentran
en la zona de alta incertidumbre (esperable) o también en zona de baja probabilidad
(problemático).

### 6.2 Análisis de falsos negativos (los más costosos)
Las transacciones fraudulentas que el modelo no detecta son las más peligrosas.
Para cada FN:
- ¿Tiene score justo por debajo del threshold?
- ¿Comparte características con otros FN?
- ¿Hay un cluster de FN no cubierto por el modelo?

### 6.3 Análisis de falsos positivos
Las alertas falsas erosionan la confianza del equipo de fraude. Para cada FP:
- ¿Tiene score justo por encima del threshold?
- ¿Hay un segmento con FP desproporcionados?
- Si es así, ¿se puede añadir una regla de negocio post-modelo para filtrarlos?

### 6.4 Curva de error acumulado por score threshold
Para cada valor de threshold, calcular qué fracción del total de errores (FP+FN)
se concentra en transacciones con score en [threshold - 0.05, threshold + 0.05].
Un pico estrecho indica que el error es sensible al threshold.

### 6.5 Análisis de concordancia LGB vs. XGB
Comparar dónde coinciden y dónde difieren LightGBM y XGBoost:
- **Coincidencia correcta**: ambos aciertan
- **Coincidencia incorrecta**: ambos fallan (el ensamble no ayuda aquí)
- **Discrepancia**: uno acierta, otro falla (el ensamble puede mejorar)
Mapear las discrepancias por feature para ver si hay patrones.

---

## 7. Implementación práctica

```python
# Flujo recomendado para análisis de residuos

cargar_modelo()
generar_lote()  # o cargar datos reales

# Pipeline de inferencia
X_fe = fe.transform(df)
y_prob = ensemble_predict(X_fe)
y_pred = (y_prob >= best_t).astype(int)

# 1. Residuos básicos
residuo_crudo = y_true - y_pred       # {-1, 0, +1}
residuo_prob  = y_true - y_prob       # [-1, +1]
pearson       = residuo_prob / np.sqrt(y_prob * (1 - y_prob) + 1e-15)

# 2. Matriz de confusión por segmento
for segmento in segmentos:
    idx = df[segmento] == valor
    tp = ((y_pred == 1) & (y_true == 1))[idx].sum()
    fn = ((y_pred == 0) & (y_true == 1))[idx].sum()
    # ... precision, recall por segmento

# 3. Calibration plot
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

# 4. Drift (PSI)
def psi(expected, actual, bins=10):
    # Population Stability Index
    ...

# 5. Análisis de errores por feature
for col in num_features:
    df['bin'] = pd.qcut(df[col], q=10, duplicates='drop')
    error_rate = df.groupby('bin').apply(
        lambda g: ((y_pred[g.index] != y_true[g.index]).mean())
    )
```

---

## 8. Cuándo usar cada técnica

| Situación | Técnica recomendada |
|-----------|-------------------|
| El modelo acaba de entrenarse | Calibración, segmentos, matriz de confusión |
| El modelo lleva semanas en producción | Drift (PSI, KS), media móvil de residuos |
| Los FP son muy altos | Análisis de FP por segmento, curve de error por threshold |
| Los FN son muy altos | Análisis de FN, cluster de errores, concordancia LGB/XGB |
| Se incorpora un segmento nuevo | Matriz de confusión segmentada |
| Se quiere monitorizar continuamente | Dashboard con media móvil de precisión/recall, PSI, residuos |
