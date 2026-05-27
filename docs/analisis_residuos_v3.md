# Análisis de Residuos — modelo_10_v3
**NovaPay ML · Mayo 2026**

---

## 1. Contexto

Modelo evaluado: `modelo_10_v3.pkl` entrenado sobre `dataset_fraude_v3.csv` (100.000 transacciones, 3,53% fraude).  
Test set: 20.000 transacciones (split 60/20/20, `random_state=42`), de las cuales 707 son fraudes.

Pipeline: FE v4 → StandardScaler → SimpleImputer(median) → LGB (w=0.65) + XGB (w=0.35) → threshold F2 = 0.7622.

---

## 2. Métricas globales

| Métrica | Valor |
|---|:---:|
| PR-AUC | 0.8848 |
| AUC-ROC | 0.9831 |
| Recall | 84.3% |
| Precision | 75.2% |
| F2 | 0.8230 |
| TP | 596 |
| FP | 197 |
| FN | 111 |
| TN | 19.096 |

El AUC-ROC (0.983) indica que el modelo tiene una capacidad de ranking excelente. La brecha entre AUC-ROC y PR-AUC (0.885) refleja el desbalanceo del 3,5% — es esperable y no indica un problema.

---

## 3. Falsos negativos por patrón

| Patrón | Total en test | FN | % no detectado | Prob media |
|---|:---:|:---:|:---:|:---:|
| burst | 170 | 1 | **0.6%** | 0.989 |
| ambos | 21 | 3 | 14.3% | 0.905 |
| clásico | 444 | 75 | 16.9% | 0.875 |
| **silencioso** | **72** | **32** | **44.4%** | **0.658** |

### 3.1 Burst — casi resuelto

Solo 1 FN sobre 170 casos. La probabilidad media de los burst detectados es 0.989 — el modelo los identifica con alta confianza. Las features `burst_rapido`, `txn_intensity` y `txn_por_minuto` están funcionando correctamente. El único FN es probablemente un caso marginal con características mixtas.

### 3.2 Clásico — margen de mejora moderado

75 FN sobre 444 casos (16.9%). La probabilidad media es 0.875, lo que indica que muchos de estos fraudes están cerca del umbral — bajando ligeramente el threshold se recuperarían a costa de más FP. No hay un patrón estructural claro: el modelo sabe que son sospechosos pero no lo suficiente para superar el 0.762.

### 3.3 Silencioso — punto ciego crítico

32 FN sobre 72 casos (44.4%). **Este es el problema principal.**

La probabilidad media de estos FN es 0.658 — muy lejos del umbral. No es que el modelo esté indeciso: genuinamente no ve la señal. El perfil de estos FN:

- Importe mediano: ~65€ (primer cuartil de los FN totales)
- Velocidad: 2-4 txns/hora (no activa `burst_rapido`)
- Dispositivo reconocido: ~65% (no activa `foreign_unknown_device`)
- Cross-border: 0% (no activa señales geográficas)
- Destino alto riesgo: 0% (no activa la señal más potente)

Son transacciones que no activan ninguna señal clásica. El modelo las interpreta como legítimas porque son indistinguibles en términos de features individuales.

---

## 4. Distribución de probabilidades en los FN

| Rango de probabilidad | Nº FN |
|---|:---:|
| < 0.30 | 38 |
| 0.30 – 0.50 | 24 |
| 0.50 – 0.60 | 26 |
| 0.60 – 0.70 | 14 |
| 0.70 – 0.762 | 9 |
| > 0.762 (no debería haber) | 0 |

El 55% de los FN tiene probabilidad menor de 0.50 — el modelo no tiene ninguna duda de que son legítimas. Estos no se recuperarían bajando el threshold sin asumir un número de FP inasumible. Necesitan nuevas features.

---

## 5. Perfil de los falsos positivos

197 FP con las siguientes características:

- **Dispositivo no reconocido**: 76% (vs 24% en legítimas del test)
- **Cross-border**: 57%
- **Destino alto riesgo**: 32%
- **Probabilidad media**: 0.86

Son transacciones legítimas que activan múltiples señales de fraude clásico simultáneamente. El modelo es coherente — no hay ruido ni errores de cálculo. La causa raíz es que en el dataset sintético la combinación cross-border + dispositivo no reconocido tiene una fraud rate alta pero no del 100%, y el modelo está ajustado para maximizar F2 (priorizando recall sobre precision).

**Por tipo de transacción:**
- Tarjeta: 108 FP (55%)
- Transferencia: 56 FP (28%)
- Bizum: 33 FP (17%)

Bizum tiene una tasa de FP proporcionalmente alta respecto a su volumen — merece monitorización en producción.

---

## 6. Diagnóstico y causa raíz

### ¿Por qué el fraude silencioso sigue siendo el punto ciego?

El modelo tiene todas las features individuales necesarias (`importe_transaccion`, `dispositivo_reconocido`, `cross_border`, `numero_transacciones_ultima_hora`), pero el fraude silencioso se define precisamente por **no activar ninguna de ellas**. La única señal posible es contextual: ¿es esta transacción anómala para *este cliente en particular*?

`diff_importe_signed` (z-score del importe respecto al perfil mensual del cliente) va en la dirección correcta pero es insuficiente sola porque:
1. Un cliente con alta variabilidad mensual tiene `desviacion_estandar_mensual` grande → el z-score es bajo incluso para importes inusuales.
2. No captura la frecuencia: 3 transacciones de 80€ en 20 minutos es más sospechoso que una transacción de 80€ aislada.

---

## 7. Recomendaciones para el siguiente ciclo

### Prioritario

**`diff_tiempo_cliente`** — z-score del tiempo entre transacciones por cliente.  
Un cliente que normalmente tiene 6 horas entre operaciones y de repente hace 3 en 20 minutos tiene un z-score muy alto aunque la velocidad absoluta (3 txns/hora) no active `burst_rapido`.

```python
# Cálculo durante fit():
self._client_time_between_means = (
    df.sort_values('fecha_hora')
    .groupby('id_cliente')['tiempo_desde_ultima_transaccion']
    .mean().to_dict()
)
self._client_time_between_stds = (
    df.sort_values('fecha_hora')
    .groupby('id_cliente')['tiempo_desde_ultima_transaccion']
    .std().fillna(1).to_dict()
)

# En transform():
mean_t = X['id_cliente'].map(self._client_time_between_means).fillna(X['tiempo_desde_ultima_transaccion'].mean())
std_t  = X['id_cliente'].map(self._client_time_between_stds).fillna(1).clip(lower=1)
X['diff_tiempo_cliente'] = (X['tiempo_desde_ultima_transaccion'] - mean_t) / std_t
```

Esta feature detectaría el fraude silencioso porque aunque 3 txns/hora no parece mucho en absoluto, puede ser 5 desviaciones estándar por encima del comportamiento habitual del cliente.

### Secundario

- **Ampliar el patrón silencioso en el dataset**: el 44% de FN indica que los 150 casos inyectados son insuficientes para que el modelo aprenda la frontera de decisión. Para la ronda 3 se recomienda aumentar a 300-400 casos silenciosos con mayor variedad de importes (incluir algunos en rango 300-500€).
- **Recalibración isotónica**: el ECE de 0.0616 es aceptable pero mejorable. Una calibración isotónica sobre el conjunto de validación podría reducirlo por debajo de 0.05.
- **Umbral adaptativo para Bizum**: el canal Bizum muestra FP proporcionalmente altos. Evaluar si el threshold de 0.783 es el óptimo o si hay margen de ajuste.

---

## 8. Comparativa con análisis v2

| Aspecto | v2 (15% fraude) | v3 (3.5% fraude) |
|---|---|---|
| Punto ciego principal | `numero_transacciones_ultima_hora` plana | Fraude silencioso (44% FN) |
| Burst detection | No existía la feature | Casi perfecto (0.6% FN) |
| FN cerca del umbral | Mayoría recuperables bajando threshold | 55% muy lejos del umbral (prob < 0.50) |
| FP perfil | Cross-border + dispositivo | Igual, + Bizum como canal nuevo |
| Señal faltante | Features de velocidad | Contexto temporal por cliente (`diff_tiempo_cliente`) |
