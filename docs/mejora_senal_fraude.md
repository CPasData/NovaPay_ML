# Mejora de Señal en Datos Sintéticos de Fraude

## Problema Original

El generador original (`generar_dataset_fraude.py`) producía datos con señal muy débil
(PR-AUC ~0.35, AUC-ROC ~0.73) porque:

1. **Características independientes**: todas las variables se generaban de forma
   independiente y con la misma distribución para todas las transacciones. La única
   dependencia entre features y la etiqueta `IS_FRAUD` venía de una fórmula de
   probabilidad aditiva con efectos pequeños (1-25%).

2. **Efectos aditivos lineales**: cada factor de riesgo contribuía individualmente
   con 2-25% de probabilidad. No había diferencias reales en la distribución de
   features entre fraude y no-fraude: un `importe_transaccion` elevado era igual
   de probable en una transacción fraudulenta que en una legítima, solo cambiaba
   la probabilidad de que esa transacción fuera etiquetada como fraude.

3. **Señal diluida**: con 20+ factores de riesgo contribuyendo cada uno 2-25%,
   la probabilidad media era ~8-10% con una desviación estándar baja. Las transacciones
   con mayor probabilidad apenas alcanzaban 40-50%, y eran extremadamente raras.

**Resultado en modelo LightGBM**: PR-AUC 0.35, AUC-ROC 0.73, F2 con recall 79% y
precisión solo 24%.

## Estrategia de Mejora

En lugar de modificar la fórmula de probabilidad (enfoque puramente aditivo, limitado
por construcción), se adoptó una estrategia de **inyección de señal post-etiquetado**:

```
1. Generar características idénticas al original v1
2. Calcular probabilidad base (misma fórmula que v1)
3. Asignar etiqueta IS_FRAUD según esa probabilidad (~15% tasa)
4. SI es fraude → modificar características post-hoc para reflejar
   comportamiento fraudulento real
5. SI no es fraude → dejar características intactas
```

Este enfoque crea **distribuciones genuinamente diferentes** entre las clases:

| Feature | No-fraude (distribución) | Fraude (distribución) |
|---|---|---|
| `dispositivo_reconocido=0` | 15% de los casos | 60% de los casos |
| `operacion_pais != customer_country` | 10% de los casos | 55% de los casos |
| `estado_tarjeta` robada/extraviada | 7% de los casos | 35% de los casos |
| `numero_transacciones_ultima_hora` | Poisson(2) | ~6-20 (alta velocidad) |
| `numero_pin_disponibles=0` | 2% de los casos | 35% de los casos |

## Modificaciones Post-Hoc Aplicadas a Transacciones Fraudulentas

Cada modificación se aplica con una probabilidad independiente (no todas se aplican
siempre, creando variabilidad realista):

| Modificación | Prob. en fraude | Efecto en detección |
|---|---|---|
| Cambiar país de operación | 55% | Alta |
| Marcar dispositivo como no reconocido | 60% | Alta |
| Marcar tarjeta como robada/extraviada | 35% | Muy alta |
| Aumentar velocidad de tx (6-20/hora) | 45% | Alta |
| Ajustar importe cerca del límite (85-99%) | 40% | Media |
| Poner PIN disponibles = 0 | 35% | Alta |
| Usar autenticación débil (firma/3DS) | 35% | Media |
| Marcar destino como alto riesgo | 30% | Alta |
| Volumen saliente >> entrante (4-8×) | 25% | Media |
| Tiempo desde última tx muy corto (<55s) | 30% | Media |

## Resultados

| Métrica | v1 (original) | v2 | Mejora |
|---|---|---|---|
| PR-AUC | 0.3466 | 0.9557 | +176% |
| AUC-ROC | 0.7316 | 0.9862 | +35% |
| Best F2 precision | 23.8% | 85.9% | +261% |
| Best F2 recall | 79.4% | 90.9% | +14% |
| Tasa de fraude | 15.3% | 15.0% | Estable |

## Validación Realista

Aunque la PR-AUC de 0.96 es artificialmente alta (no esperable en datos reales),
es intencional para:

1. **Validar el pipeline extremo a extremo**: selección de modelo, optimización de
   hiperparámetros, selección de threshold, calibración, etc.
2. **Asegurar decisiones estables**: con señal fuerte, las decisiones de umbral
   son reproducibles y no dominadas por ruido de muestreo.
3. **Comparar modelos**: con PR-AUC 0.96, diferencias entre LightGBM, XGBoost y
   otros modelos son significativas y no ruido.

Al llegar a datos reales, el desempeño se recalibra a la baja, pero el pipeline
ya está validado.

## Cómo Usar

```bash
python scripts/synthetic_data/generar_dataset_fraude_v2.py

Esto produce `data/dataset_fraude_v2.csv` con las **mismas columnas** que el
original.
