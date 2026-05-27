# NovaPay ML — Análisis de Residuos y Recomendaciones
**Consolidado equipo Data Science — Mayo 2026**

---

## 1. Introducción

Este documento consolida el análisis de residuos y evaluación del modelo de detección de fraude de NovaPay, integrando los hallazgos empíricos obtenidos del análisis estadístico del dataset v2 con el marco metodológico de evaluación propuesto por el equipo. El objetivo es proporcionar un mapa claro de qué funciona, qué falla y qué mejoras concretas deben incorporarse en el próximo ciclo de reentrenamiento.

El modelo actual es un ensemble LightGBM + XGBoost (30%/70%) entrenado sobre el dataset sintético v2 con señal reforzada, optimizado con umbral F2 en 0.314 y PR-AUC de 0.9557 sobre datos de entrenamiento. El siguiente reto es mantener esa capacidad discriminativa frente a los nuevos patrones de fraude generados por el equipo de Ciberseguridad en las rondas sucesivas.

---

## 2. Señales Discriminativas Confirmadas

El análisis empírico del dataset identificó las siguientes señales con mayor poder discriminativo entre transacciones legítimas y fraudulentas. Estas cifras proceden del análisis de distribuciones sobre el conjunto completo antes de la partición train/test.

### 2.1 Top de features por separación de clases

| Feature | Fraudes | Legítimas | Poder discriminativo |
|---|---|---|---|
| `cross_border` | 64.0% | 8.2% | Muy alto |
| `foreign_unknown_device` (flag compuesto) | 43.6% | 1.0% | Muy alto |
| `dispositivo_reconocido = 0` | 47.2% | 6.0% | Muy alto |
| `destino_alto_riesgo` | ~30% | ~3% | Alto |
| `burst_rapido` (>5 tx/hora, <5min) | ~45% | ~5% | Alto |
| `outflow_inflow_ratio` (mediana) | 3.42 | 1.01 | Alto |
| `net_flow_30d` (mediana) | −2.163 € | −19 € | Alto |
| `is_night` | 16.2% | 14.6% | Muy bajo |
| `is_weekend` | 15.1% | 15.0% | Irrelevante |

### 2.2 Gradiente de acumulación de flags

El modelo se beneficia de la combinación de flags de riesgo. El análisis muestra un gradiente muy pronunciado:

| Nº de flags activos | Tasa de fraude observada |
|---|---|
| 0 flags | 0.5% |
| 1 flag | ~5% |
| 2 flags | 24.1% |
| 3 flags | ~55% |
| 4 flags | 92.5% |
| 5+ flags | ~99% |
| 6 flags | 100% |

Este gradiente valida que el modelo no depende de una única señal, sino de la co-ocurrencia de múltiples indicadores de riesgo. Es un comportamiento deseable y robusto frente a ataques que intenten evadir una sola regla.

### 2.3 Features con señal débil o nula

`is_night` (16.2% en fraudes vs 14.6% en legítimas) e `is_weekend` (15.1% vs 15.0%) muestran prácticamente ninguna capacidad discriminativa de forma aislada. No se propone eliminarlas porque pueden contribuir en combinación dentro de los árboles, pero no deben considerarse señales primarias.

---

## 3. Features Problemáticas

### 3.1 `high_ratio_redondeado` — eliminar en el próximo reentrenamiento

Esta feature combina dos condiciones: que el importe supere el 85% del límite de la tarjeta **Y** que el importe sea múltiplo exacto de 100. La intención era capturar el patrón de tarjetas probadas con importes redondos cerca del límite.

**Problema:** solo aparece en el 0.3% de los fraudes del dataset. La condición del importe redondo destruye la señal del ratio, que por sí sola sí tiene valor discriminativo.

`txn_vs_limit_pct` ya está presente como feature independiente, por lo que la señal del ratio alto no se pierde al eliminar `high_ratio_redondeado`. Solo se elimina la restricción del importe redondo, que es la que inutiliza el flag.

> **Nota de implementación:** no se puede eliminar del código sin reentrenar el modelo, ya que el pipeline guardado en el `.pkl` fue entrenado con esta feature. La eliminación debe hacerse en el próximo ciclo de reentrenamiento de forma coordinada.

---

## 4. Mejoras Propuestas para Feature Engineering

### 4.1 Mejorar `diff_importe_cliente` con z-score real

La implementación actual calcula:

```
|importe – media_cliente| / media_cliente
```

Tiene dos limitaciones. Primero, ignora la variabilidad del cliente: un cliente que siempre paga exactamente 50 € y otro que varía entre 10 € y 2.000 € reciben el mismo tratamiento, cuando una transacción de 500 € es alarmante para el primero y completamente normal para el segundo. Segundo, sufre del problema de cold start: para clientes no vistos en entrenamiento, el mapa devuelve NaN y el valor se imputa a 0, haciéndolo invisible para el modelo.

**Propuesta:** el dataset ya incluye `importe_medio_mensual` y `desviacion_estandar_mensual` como campos directos de cada transacción. Esto permite calcular un z-score real sin depender del fit:

```python
# Cuántas desviaciones típicas se aleja del comportamiento habitual del cliente
diff_importe_zscore = |importe – importe_medio_mensual| / desviacion_estandar_mensual

# Versión con signo: positivo = transacción más grande de lo normal (patrón fraude)
diff_importe_signed = (importe – importe_medio_mensual) / desviacion_estandar_mensual

# Flag binario para anomalías extremas
importe_anomalo = 1 si diff_importe_zscore > 3
```

La versión con signo es especialmente útil porque los fraudes tienden a ser importes superiores a lo habitual para ese cliente. El z-score positivo alto es una señal más directa que el valor absoluto.

---

## 5. Optimización del Threshold

### 5.1 Limitación del threshold único global

El modelo aplica actualmente un umbral único de 0.314 a todas las transacciones, independientemente del canal. Esto genera una compensación imposible: si se sube el umbral para no bloquear pagos con tarjeta legítimos, se pierden fraudes en transferencias. Si se baja para cazar transferencias sospechosas, se bloquean demasiadas operaciones con tarjeta.

### 5.2 Threshold por canal (`tipo_transaccion`)

Cada canal tiene un perfil de riesgo distinto. La propuesta es calibrar un umbral específico para cada `tipo_transaccion`, optimizando F2-score por separado sobre el subconjunto correspondiente del dataset:

| Canal | Dirección del umbral | Justificación |
|---|---|---|
| `tarjeta` | Más conservador (~0.38) | Mayor volumen de transacciones legítimas. Los falsos positivos tienen mayor impacto en UX. |
| `transferencia` | Más agresivo (~0.25) | Mayor riesgo unitario. Los falsos negativos son más costosos que bloquear una transferencia. |
| `bizum` | Intermedio (~0.30) | Importes generalmente menores. Balance entre precisión y recall. |

> **Condición:** para que los umbrales por canal sean robustos se necesitan suficientes ejemplos de fraude por canal en el dataset de entrenamiento. Si un canal tiene muy pocos fraudes, el umbral optimizado será inestable.

### 5.3 Sensibilidad del threshold

Para cada valor de threshold, se puede calcular qué fracción del total de errores (FP+FN) se concentra en transacciones con score en [threshold − 0.05, threshold + 0.05]. Un pico estrecho indica que el error es muy sensible a pequeños cambios en el umbral, lo que hace el sistema frágil. Un perfil plano indica robustez. Este análisis debe ejecutarse tras cada reentrenamiento.

---

## 6. Evaluación Avanzada del Modelo

### 6.1 Tipos de residuos en clasificación binaria

| Tipo | Fórmula | Propósito |
|---|---|---|
| Residuo crudo | `y_true − y_pred ∈ {−1, 0, +1}` | −1 = FP, 0 = acierto, +1 = FN |
| Residuo de probabilidad | `y_true − y_prob ∈ [−1, +1]` | Magnitud del error de probabilidad |
| Residuo de Pearson | `(y_true − y_prob) / √(y_prob·(1−y_prob))` | Error estandarizado por varianza esperada |
| Residuo de deviance | `sign(y_true−y_prob) · √(−2·log-likelihood)` | Contribución a la pérdida total |

### 6.2 Análisis de errores por segmento

El modelo puede tener rendimientos muy dispares entre segmentos aunque su métrica global sea buena. Se recomienda calcular precisión, recall y PR-AUC dentro de cada subconjunto: `tipo_transaccion`, `metodo_autenticacion`, país doméstico vs transfronterizo, rango de importe y franja horaria.

Si un segmento muestra recall significativamente menor que la media global, el modelo tiene un sesgo sistemático hacia ese segmento. Es especialmente relevante para detectar qué tipos de fraude el modelo está ignorando de forma consistente.

### 6.3 Análisis de falsos negativos — los más costosos

Las transacciones fraudulentas no detectadas (FN) son las de mayor impacto económico. El análisis debe responder tres preguntas:

- **Proximidad al umbral:** ¿tiene score justo por debajo del threshold? Si es así, un ajuste fino del umbral los recupera.
- **Clustering:** ¿comparten características comunes? Si existe un cluster de FN con un patrón coherente, el modelo tiene un punto ciego estructural.
- **Concordancia entre modelos:** ¿falla solo LightGBM, solo XGBoost, o ambos? Si ambos fallan en los mismos casos, el ensemble no aporta y hay que incorporar nueva información.

> Los nuevos patrones de fraude generados por Ciber en cada ronda aparecerán primero como FN. Analizar sus características antes de reentrenar permite diseñar features más dirigidas.

### 6.4 Análisis de falsos positivos — erosión de confianza

Los FP son transacciones legítimas bloqueadas. Si un tipo de operación genera muchos FP, puede añadirse una regla de negocio post-modelo para filtrarlos sin afectar al resto. Los FP con score justo por encima del umbral son recuperables con ajuste de threshold. Los que tienen score alto son casos donde el modelo está genuinamente confundido y requieren features adicionales.

### 6.5 Análisis de concordancia LightGBM vs XGBoost

| Caso | Descripción | Implicación |
|---|---|---|
| Coincidencia correcta | Ambos aciertan | El ensemble es sólido aquí |
| Coincidencia incorrecta | Ambos fallan | El ensemble no ayuda. Requiere nuevas features |
| Discrepancia | Uno acierta, otro falla | El ensemble añade valor. Se puede recalibrar el peso |

Si la mayoría de los FN son «coincidencia incorrecta», los pesos del ensemble no son el problema: el modelo necesita información que no tiene. Si hay muchas discrepancias, los pesos actuales (30/70) pueden estar desaprovechando la diversidad de ambos modelos.

---

## 7. Calibración del Modelo

Un modelo bien calibrado produce probabilidades que se corresponden con las frecuencias reales de fraude: si el modelo dice 0.8, el 80% de esas transacciones deberían ser fraude. La calibración es especialmente importante cuando se usa la probabilidad para priorizar revisiones manuales.

| Métrica | Cómo interpretar | Umbral aceptable |
|---|---|---|
| Brier Score | Media de (y_true − y_prob)². Combina calibración y discriminación. | < 0.10 |
| ECE (Expected Calibration Error) | Diferencia media absoluta entre prob. predicha y frecuencia observada. | < 0.05 |
| Reliability Diagram | Los puntos deben estar sobre la diagonal. Desviación sistemática → recalibrar. | Visual |

Con el dataset sintético v2 es esperable que la calibración sea buena en entrenamiento. El riesgo real es la descalibración tras el primer reentrenamiento con datos reales. Medir Brier Score y ECE tras cada ciclo permite detectarlo a tiempo.

---

## 8. Monitorización en Producción

Una vez el modelo esté en producción, el rendimiento puede degradarse por concept drift (los patrones de fraude cambian) o data drift (las distribuciones de entrada cambian). Ambos requieren mecanismos de detección distintos.

### 8.1 Drift en scores de probabilidad (PSI)

El Population Stability Index compara la distribución de probabilidades predichas del lote actual frente al lote de entrenamiento:

| Valor PSI | Interpretación | Acción |
|---|---|---|
| < 0.10 | Distribución estable | Sin acción |
| 0.10 – 0.25 | Cambio moderado | Investigar |
| > 0.25 | Cambio significativo | Evaluar reentrenamiento |

### 8.2 Drift en residuos

Si la media móvil del residuo de probabilidad se vuelve sistemáticamente positiva (más FN), el modelo está perdiendo fraudes que antes captaba. Si se vuelve sistemáticamente negativa (más FP), está siendo demasiado conservador. El KS-test entre la distribución de scores actual y el baseline de entrenamiento complementa este análisis.

### 8.3 Drift en features

Se recomienda monitorizar con KS-test (numéricas) y chi-cuadrado (categóricas) entre el lote actual y la distribución de referencia. Especial atención a `cross_border`, `dispositivo_reconocido` y `burst_rapido`, que son los mayores discriminadores.

> Cada nueva ronda de ataques de Ciberseguridad introduce nuevos patrones que aparecerán como drift en features. El sistema de monitorización es la primera línea de detección de que el modelo necesita reentrenarse.

---

## 9. Resumen de Recomendaciones Priorizadas

| # | Recomendación | Tipo | Prioridad | Cuándo |
|---|---|---|---|---|
| 1 | Eliminar `high_ratio_redondeado` del feature engineering | Limpieza | Alta | Próximo reentrenamiento |
| 2 | Sustituir `diff_importe_cliente` por z-score real con `importe_medio_mensual` y `desviacion_estandar_mensual` | Mejora FE | Alta | Próximo reentrenamiento |
| 3 | Implementar threshold por canal (`tipo_transaccion`) | Optimización | Alta | Próximo reentrenamiento |
| 4 | Análisis de FN por cluster en cada ronda de Ciber | Evaluación | Alta | Tras cada ronda |
| 5 | Análisis de concordancia LGB vs XGB post-reentrenamiento | Evaluación | Media | Tras reentrenamiento |
| 6 | Medir Brier Score y ECE tras cada reentrenamiento | Calibración | Media | Continuo |
| 7 | Implementar PSI y media móvil de residuos en producción | Monitorización | Media | Despliegue |
| 8 | Análisis de FP por segmento para reglas post-modelo | Evaluación | Baja–Media | Producción |

---

*Las cifras empíricas de este documento proceden del análisis del dataset sintético v2 (`dataset_fraude_mejorado.csv`, PR-AUC 0.9557). Los valores no son representativos de datos reales de producción, donde las distribuciones serán más ruidosas y el rendimiento esperado será menor.*
