# Análisis Comparativo de Residuos — modelo_09_v3 vs modelo_11_v3
**NovaPay ML · Mayo 2026**

---

## 1. Contexto

Comparativa entre el modelo de referencia del equipo (`modelo_09_v3.pkl`) y el modelo retrenado con features adicionales (`modelo_11_v3.pkl`).

| | modelo_09_v3 | modelo_11_v3 |
|---|---|---|
| Autor | Compi | Nuestro |
| Features | 69 | 71 (+`burst_cross_border`, +`cross_border_high_risk`) |
| Peso ensemble | LGB 65% + XGB 35% | LGB 76% + XGB 24% |
| Threshold global | 0.7622 | 0.7641 |
| Thresholds por canal | tarjeta=0.759, transf=0.724, bizum=0.783 | tarjeta=0.739, transf=0.763, bizum=0.896 |

Test set: 20.000 transacciones (split 80/20, `random_state=42`, estratificado), de las cuales **707 son fraudes** (3.54%).  
Pipeline común: FE → StandardScaler → SimpleImputer(median) → LGB + XGB → threshold por canal.

---

## 2. Métricas globales

| Métrica | modelo_09 | modelo_11 | Δ |
|---|:---:|:---:|:---:|
| PR-AUC | 0.8784 | **0.8883** | +0.0099 |
| AUC-ROC | 0.9831 | **0.9847** | +0.0016 |
| Recall | **86.7%** | 83.5% | −3.2pp |
| Precision | 60.6% | **77.8%** | +17.2pp |
| F2-score | 0.7984 | **0.8226** | +0.024 |
| Brier score | 0.0414 | **0.0255** | −0.016 |
| ECE | 0.0922 | **0.0592** | −0.033 |
| TP | 613 | 590 | −23 |
| FP | 398 | **168** | **−230** |
| FN | **94** | 117 | +23 |
| TN | 18.895 | 19.125 | +230 |

El modelo_11 gana en F2 (+0.024) a pesar de sacrificar 23 detecciones, porque la reducción de 230 falsos positivos tiene un peso desproporcionado en la métrica. El AUC-ROC y PR-AUC confirman que no es solo un efecto de umbral: el modelo_11 clasifica mejor en toda la curva.

---

## 3. Falsos positivos — la mejora principal

**modelo_09: 398 FP → modelo_11: 168 FP (−230, −57.8%)**

Son 230 transacciones legítimas menos que el sistema hubiera bloqueado o enviado a revisión manual innecesariamente. Es el cambio más significativo de esta iteración.

### 3.1 Perfil de los FP

| Característica | m09 (n=398) | m11 (n=168) |
|---|:---:|:---:|
| Dispositivo no reconocido | 68.6% | 70.8% |
| Cross-border | 55.3% | 50.0% |
| Destino alto riesgo | 34.7% | 33.3% |
| Cross-border + alto riesgo | 16.1% | **20.8%** |
| Prob. media | 0.857 | 0.867 |

El perfil de los FP que quedan en modelo_11 es prácticamente el mismo — dispositivo no reconocido + señales geográficas — pero en menor cantidad. La feature `cross_border_high_risk` aparece en el 20.8% de los FP supervivientes, indicando que el modelo sí la usa como señal, aunque no de forma perfecta (hay transacciones legítimas cross-border con destinos de alto riesgo).

### 3.2 FP por canal

| Canal | Txns | FP m09 | % m09 | FP m11 | % m11 | Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Tarjeta | 12.142 | 228 | 1.88% | 117 | 0.96% | −111 |
| Transferencia | 5.011 | 118 | 2.35% | 41 | 0.82% | −77 |
| Bizum | 2.847 | 52 | 1.83% | **10** | **0.35%** | −42 |

La mejora más llamativa es en Bizum, donde el threshold del modelo_11 (0.896 vs 0.783 del modelo_09) es sensiblemente más conservador — el modelo exige mucha más evidencia para detectar fraude en este canal. Esto explica casi en su totalidad la caída de FP en Bizum: −42 FP (−80.8%).

---

## 4. Falsos negativos por patrón

Distribución de fraudes en el test set: 217 burst (30.7%), 458 clásico (64.8%), 32 silencioso (4.5%).

| Patrón | Fraudes | FN m09 | % m09 | FN m11 | % m11 | Δ FN | Prob FN m09 | Prob FN m11 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Burst | 217 | 1 | 0.5% | 5 | **2.3%** | +4 | 0.105 | 0.482 |
| Clásico | 458 | 79 | 17.2% | 93 | 20.3% | +14 | 0.416 | 0.406 |
| **Silencioso** | 32 | **14** | **43.8%** | **19** | **59.4%** | +5 | 0.379 | 0.385 |

### 4.1 Burst — sigue siendo casi perfecto

El modelo_09 solo falla 1 de 217 bursts; el modelo_11 falla 5. Ambos son excelentes. El ligero empeoramiento en modelo_11 se debe al mayor peso LGB (0.76 vs 0.65) — LightGBM es algo más conservador en este patrón que XGBoost, pero la detección sigue siendo del 97.7%.

### 4.2 Clásico — regresión moderada y esperada

14 FN adicionales (17.2% → 20.3%). Son transacciones que activan señales como cross-border o destino alto riesgo, pero el modelo_11 requiere mayor evidencia para cruzar el umbral. La probabilidad media de los FN apenas cambia (0.416 → 0.406), lo que indica que el modelo los ve igual de "sospechosos pero no suficientes" — no hay una pérdida de capacidad de ranking.

### 4.3 Silencioso — punto ciego que empeora

5 FN adicionales (43.8% → 59.4%). Este es el resultado más preocupante del comparativo.

El perfil silencioso se define por no activar ninguna señal clásica: dispositivo reconocido, sin cross-border, sin destino de alto riesgo, sin burst. Ninguna de las dos features nuevas (`burst_cross_border`, `cross_border_high_risk`) ayuda en este patrón — son interacciones de señales que, por definición, el silencioso no tiene. La regresión tiene una causa más sutil: el mayor peso LGB en modelo_11 (0.76) perjudica levemente el silencioso porque LightGBM, entrenado con más presión hacia la precision, es más exigente con la evidencia contextual.

La probabilidad media apenas cambia (0.379 → 0.385), confirmando que el problema no es el umbral sino la capacidad discriminativa: el modelo no ve la señal.

---

## 5. Distribución de probabilidades en los FN

| Rango | FN m09 | FN m11 |
|---|:---:|:---:|
| < 0.30 | 30 | 42 |
| 0.30 – 0.50 | 26 | 29 |
| 0.50 – 0.60 | 17 | 14 |
| 0.60 – 0.70 | 11 | 19 |
| 0.70 – thr | 10 | 13 |

El modelo_11 tiene más FN con probabilidad muy baja (<0.30): 42 vs 30. Esto refuerza el diagnóstico del fraude silencioso — son transacciones que el modelo_11 clasifica con más confianza como legítimas (no dudas cercanas al umbral). No se recuperarían simplemente bajando el threshold.

---

## 6. Calibración

El ECE (Expected Calibration Error) mide cuánto se desvían las probabilidades predichas de las frecuencias reales:

- modelo_09: ECE = 0.0922 → la probabilidad media en muchos bins difiere casi un 9% de la tasa real. Por ejemplo, si el modelo dice 0.80, la tasa real de fraude en esas transacciones es ≈0.71.
- modelo_11: ECE = 0.0592 → desviación de ~6%. Mejor, pero todavía por encima del objetivo de 0.05.

La mejor calibración del modelo_11 proviene de la combinación de mayor peso LGB (histgrambased trees tienden a calibrar mejor que XGBoost puro) y del reentrenamiento completo sobre el mismo dataset. Una calibración isotónica en el conjunto de validación podría bajar el ECE por debajo de 0.05 sin reentrenar el modelo base.

---

## 7. Diagnóstico: ¿por qué mejora la precision pero empeora el silencioso?

Las dos features nuevas actúan exactamente como se esperaba:

**`cross_border_high_risk`** = `cross_border AND destino_alto_riesgo`. Identifica la combinación más agresiva del fraude clásico. Esto permite al modelo distinguir mejor entre transacciones cross-border normales y cross-border de alto riesgo → menos FP en el grupo cross-border, que era el mayor grupo de FP en modelo_09.

**`burst_cross_border`** = `burst_rapido AND cross_border`. Un burst que además sale al extranjero es señal de account takeover coordinado. Mejora la confianza en el patrón burst clásico.

Ninguna de las dos features aporta información sobre el fraude silencioso porque ese patrón no activa ni burst ni cross-border ni destino alto riesgo. La regresión en silencioso es un efecto colateral del rebalanceo del ensemble (más LGB, menos XGB) y no de una pérdida de información.

---

## 8. Recomendaciones

### Prioritario — desbloquear el fraude silencioso

El único camino para mejorar el silencioso es dar al modelo información temporal por cliente: ¿es esta transacción inusual para *este cliente en particular*?

**`diff_tiempo_cliente`** — z-score del tiempo entre transacciones respecto al perfil histórico del cliente:

```python
# En fit():
self._client_time_means = df.groupby('id_cliente')['tiempo_desde_ultima_transaccion'].mean().to_dict()
self._client_time_stds  = df.groupby('id_cliente')['tiempo_desde_ultima_transaccion'].std().fillna(1).to_dict()

# En transform():
mean_t = X['id_cliente'].map(self._client_time_means).fillna(X['tiempo_desde_ultima_transaccion'].mean())
std_t  = X['id_cliente'].map(self._client_time_stds).fillna(1).clip(lower=1)
X['diff_tiempo_cliente'] = (X['tiempo_desde_ultima_transaccion'] - mean_t) / std_t
```

Un cliente que normalmente opera cada 6 horas y de repente realiza 3 transacciones en 20 minutos tiene un z-score extremo aunque la velocidad absoluta no active `burst_rapido`.

### Secundario

**Recalibración isotónica**: el ECE de modelo_11 es 0.0592, objetivo <0.05. Fit de `CalibratedClassifierCV(method='isotonic')` en el conjunto de validación (sin tocar el test) puede conseguirlo sin reentrenar.

**Ampliar patrón silencioso en el dataset**: 32 fraudes silenciosos en test (4.5% del fraude total) es insuficiente para que el modelo aprenda bien la frontera. Se recomienda aumentar a 60-80 casos silenciosos en la siguiente generación del dataset, con mayor variedad de importes (incluir rango 300-600€) y de perfiles de cliente.

**Revisar threshold de Bizum**: el salto de 0.783 → 0.896 reduce drásticamente los FP (−80.8%) pero merece monitorización en producción — si la fraud rate real de Bizum es menor que en el dataset, el threshold alto puede estar bien calibrado; si es mayor, podría estar dejando escapar fraudes.

---

## 9. Resumen ejecutivo

modelo_11 supera a modelo_09 en todos los indicadores de ranking (PR-AUC, ROC, F2, Brier, ECE) y reduce los falsos positivos en un 57.8% (−230 FP), lo que significa menos bloqueos innecesarios a clientes legítimos y menos carga para el equipo de revisión manual.

El coste es moderado: 23 fraudes adicionales no detectados (+24.5% FN), concentrados principalmente en el patrón clásico. El patrón silencioso empeora (43.8% → 59.4% de FN) pero por razones estructurales — las nuevas features no cubren ese patrón por diseño.

**Veredicto**: modelo_11 es el candidato recomendado para producción. El trabajo pendiente más importante es implementar `diff_tiempo_cliente` para atacar el punto ciego del fraude silencioso en la siguiente ronda de entrenamiento.

| | modelo_09 | modelo_11 | Ganador |
|---|:---:|:---:|:---:|
| F2 | 0.7984 | 0.8226 | modelo_11 |
| PR-AUC | 0.8784 | 0.8883 | modelo_11 |
| FP | 398 | 168 | modelo_11 |
| FN | 94 | 117 | modelo_09 |
| Calibración (ECE) | 0.0922 | 0.0592 | modelo_11 |
| Silencioso FN% | 43.8% | 59.4% | modelo_09 |
| **Overall** | | | **modelo_11** |
