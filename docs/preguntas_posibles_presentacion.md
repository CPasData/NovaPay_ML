# Posibles preguntas en la presentación — NovaPay ML

## 1. Sobre los datos sintéticos

**P: ¿Cómo sabemos que el modelo funcionará con datos reales si solo lo hemos probado con datos sintéticos?**

R: Los datos sintéticos son una aproximación controlada. Tenemos 3 niveles de realismo:
- **Estructural**: la jerarquía cliente→cuenta→tarjeta→transacción con correlaciones realistas
  (premium tiene más límite, mayor edad usa más firma, etc.)
- **Distribucional**: la inyección de señal post-hoc crea distribuciones que imitan patrones
  reales documentados (fraude cross-border = 64%, dispositivo no reconocido = 47%, ráfagas = 45%).
- **Validación**: el feedback de los analistas (target_final) se incorpora al reentrenamiento.
  El modelo empieza con datos sintéticos y se refina con datos reales progresivamente.

**El riesgo real** no es que el modelo no funcione, es que los patrones de fraude real sean
distintos a los sintéticos. Por eso el monitoreo de drift (KS-test, PSI) es parte del
despliegue desde el día 1. Si hay drift, se dispara alerta y se reentrena.

**P: ¿Por qué no usasteis datos reales del banco desde el principio?**

R: Porque al inicio del proyecto no había datos etiquetados. El equipo de Ciberseguridad
genera rondas de fraude semanales. Necesitábamos un modelo funcional desde el día 1 para
empezar a etiquetar. Los datos sintéticos nos permitieron arrancar sin esperar 6 meses de
recogida de datos. Es una estrategia estándar en fraude: synthetic-to-real transfer.

**P: La tasa de fraude del 15% en v1/v2 es ridícula, ¿no os pareció poco realista?**

R: Sí, y lo corregimos en v3 (3.5%). El 15% era un artefacto del generador original.
En v1 se puso esa tasa para tener suficientes casos positivos con 10K registros.
Cuando pasamos a 100K registros en v3, pudimos bajar al 3.5% realista manteniendo
~3.500 fraudes para entrenar. Si hubiera sido 3.5% con 10K registros, solo habríamos
tenido ~350 fraudes, insuficiente para entrenar un ensemble.

**P: ¿Cómo validáis que los datos sintéticos se parecen a los reales?**

R: Hoy no tenemos datos reales suficientes para una validación estadística rigurosa
(comparación de distribuciones con KS-test, correlaciones). Es una limitación asumida.
El plan es: tras 3 meses de producción con analistas etiquetando, tendremos ~15.000
transacciones reales etiquetadas. En ese punto haremos una validación formal
(comparación de distribuciones, PR-AUC en datos reales vs sintéticos) y ajustaremos
el generador sintético si es necesario.

**P: ¿El generador sintético cubre todos los tipos de fraude? ¿Fraude organizado, sintético (creación de cuenta falsa), etc.?**

R: Hoy cubre fraude transaccional: tarjeta robada, cuenta tomada, transferencia fraudulenta.
No cubre fraude de origen (fake account opening), fraude sintético (identidades falsas),
ni fraude interno (insider threat). Son tipos de fraude que requieren otros modelos
y otras features. Están en el roadmap para v4-v5.

---

## 2. Sobre el modelo y métricas

**P: 78.7% de precisión significa que 1 de cada 5 alertas es falso positivo. ¿Es aceptable?**

R: Depende del coste de cada falso positivo. Si un falso positivo significa llamar al
cliente para verificar una transacción, y el cliente se molesta, el coste es alto.
Si significa simplemente marcar la transacción para revisión asíncrona (el analista
la revisa en 30 segundos y la aprueba), el coste es bajo.

En nuestro caso, el falso positivo NO bloquea la transacción (depende de la política
de negocio). El analista revisa y libera. El coste es ~30 segundos de su tiempo.
Con 1.600 falsos positivos/día y 4 analistas, cada uno dedica ~25% de su tiempo
a liberar falsos positivos. Mejorable, pero asumible.

**P: 84.7% de recall significa que 15 de cada 100 fraudes se escapan. ¿Cuánto dinero se pierde?**

R: Depende del importe medio de los fraudes no detectados. Si los fraudes que se escapan
son de bajo importe (los que más se parecen a legítimos, suelen ser importes pequeños),
la pérdida es menor. Si los importes son uniformes, se pierde ~15% del valor total
fraudulento. En nuestro dataset sintético, los fraudes no detectados tienen importe
medio un ~30% menor que los detectados (los más grandes tienen señales más claras).

**Cálculo rápido**: si el fraude total estimado es 7.000 tx/día × 300€ = 2,1M€/día,
y el 84.7% se detecta → 1,78M€ bloqueados, 0,32M€ escapados. Con 4 analistas
(120.000€/año = 480€/día), el retorno es ~3.700:1. Es decir, por cada euro invertido
en analistas, se recuperan ~3.700€ en fraude bloqueado.

**P: PR-AUC 0.90 es peor que 0.96 de v2. ¿No estáis empeorando?**

R: No se puede comparar PR-AUC entre datasets con distinta tasa de fraude.
PR-AUC depende de la prevalencia: con 15% de fraude, el baseline es 0.15.
Con 3.5%, el baseline es 0.035. El PR-AUC de v3 (0.90) está mucho más cerca de 1.0
respecto a su baseline que v2 (0.96) respecto al suyo.

La métrica comparable es AUC-ROC: 0.9869 en v3 vs 0.9864 en v2. Esencialmente idéntico.
El modelo ordena igual de bien. Lo que cambia es la operativa (threshold por canal,
recall@k) y el realismo (3.5% vs 15%).

**P: ¿Por qué no usáis deep learning? Todo el mundo usa transformers ahora.**

R: En datos tabulares, los gradient-boosted trees (LightGBM, XGBoost) siguen siendo
estado del arte. Los transformers destacan en datos secuenciales (texto, series
temporales largas). Para una transacción con 44 campos tabulares, un ensemble de
trees da mejor resultado con menos datos, menos hiperparámetros y más interpretabilidad.

La literatura reciente (2023-2025) muestra que en datasets tabulares de <100K filas,
XGBoost/LightGBM superan o igualan a TabNet, FT-Transformer y otros modelos DL.
Si creciéramos a millones de transacciones con secuencias temporales largas,
revisaríamos la decisión.

**P: El ECE de 0.0616 está por encima del 0.05. ¿No es un problema?**

R: Es una debilidad conocida. Significa que las probabilidades están ligeramente
descalibradas: si el modelo dice prob=0.80, en realidad acierta ~74% de las veces.
Esto afecta a la interpretabilidad de prob_fraud para el analista.

La causa es el desbalance extremo (3.5% fraude). Está en el roadmap para v3.1:
recalibración isotónica para bajar ECE < 0.05. Es un cambio de 10 líneas de código
y un reentrenamiento. No es un problema de estructura, es un ajuste fino.

**P: El recall@k es 2.83%. Eso es terriblemente bajo.**

R: Sí, con 200 alertas/día. Pero no es culpa del modelo — es culpa de las
**restricciones operativas**. El top 0.1% de las transacciones solo puede contener
el 0.1% de los fraudes si la distribución fuera aleatoria. Que nuestro recall@k
sea 28× el azar demuestra que el modelo concentra bien los fraudes arriba.

Con 1.000 alertas/día (4 analistas), sube a 12.9%. Con 2.000 a 23.4%.
El modelo no es el cuello de botella — lo es la capacidad de revisión humana.
(Véase tabla completa en sección 4.5 del guión.)

**P: ¿Por qué no optimizáis directamente recall@k en lugar de F2? Sería más honesto.**

R: Porque recall@k no es diferenciable (tiene un argmax discontinuo), no se puede
optimizar con gradient descent directamente. Hay aproximaciones (LambdaRank, listwise
ranking) pero añaden complejidad. Nuestra estrategia es: optimizar F2-score (que da
buen ranking) y luego medir recall@k como métrica de evaluación, no de entrenamiento.

Dicho esto, XGBoost soporta `objective='rank:map'` que optimiza una aproximación
suave del recall@k. Lo exploraremos en v4 si el recall@k se vuelve prioritario.

---

## 3. Sobre decisiones de diseño

**P: ¿Por qué un ensemble LGB+XGB en lugar de un solo modelo?**

R: Dos razones:
1. **Rendimiento**: en validación, el ensemble gana ~0.02-0.03 de PR-AUC respecto
   al mejor modelo individual. LightGBM captura mejor los outliers y alta cardinalidad;
   XGBoost maneja mejor el desbalance extremo. Se complementan.
2. **Robustez**: si uno de los dos modelos falla en una región del espacio (por
   ejemplo, XGBoost no ve suficientes fraudes bizum), el otro lo compensa.
   El peso se optimiza por validación: en v3, 65% LGB + 35% XGB.

**Contra**: duplica el tiempo de inferencia (~50ms vs ~25ms), duplica la memoria
(~200MB vs ~100MB). Para 200K tx/día, 50ms es irrelevante.

**P: Los thresholds por canal son 0.759, 0.724, 0.783. Son muy parecidos. ¿Realmente merece la pena?**

R: La diferencia es pequeña pero consistente. El threshold de transferencia (0.724)
es más bajo porque el riesgo unitario es mayor. El de bizum (0.783) es más alto
porque hay menos datos y preferimos ser conservadores. La optimización por canal
nos da ~+0.01 de F2 sobre usar un threshold global para todos.

Merece la pena porque son 20 líneas de código y no añade complejidad operativa:
la API ya recibe `tipo_transaccion` en la request. Es prácticamente gratis.

**P: ¿Por qué no modeláis explícitamente la secuencia de transacciones de un cliente?**

R: Hoy tenemos features de sesión (tx última hora, burst_rapido, txn_por_minuto)
que capturan información secuencial agregada. No usamos LSTM/Transformer porque:
1. No tenemos secuencias largas etiquetadas (el dataset son transacciones sueltas)
2. Las features de sesión actuales capturan ~80% de la señal secuencial
3. Añadir secuencial explícito requeriría reestructurar el pipeline de inferencia
   (estado por cliente entre requests)

Si en el futuro se despliega un sistema de prevención en tiempo real con memoria
de sesión del cliente, sería el momento de añadirlo.

**P: ¿Por qué eliminasteis `high_ratio_redondeado`? No era costosa de mantener.**

R: Aparecía en el 0.3% de fraudes y la condición `importe % 100 == 0` destruía la
señal del ratio alto. Era ruido, no señal. `txn_vs_limit_pct` ya captura el ratio
alto sin la restricción artificial. Se eliminó siguiendo una recomendación del
análisis de residuos. No afectó a las métricas (PR-AUC se mantuvo).

**P: ¿Por qué quitáis `IS_FRAUD` de la respuesta? El analista lo necesita.**

R: No lo quitamos. Devolvemos `is_fraud` (0/1) que es la decisión del modelo.
`IS_FRAUD` (mayúsculas, del dataset) es la etiqueta real que usamos para entrenar.
En producción no existe — no sabemos si una transacción es fraude hasta que el
analista lo confirma. Por eso generamos datasets "sin etiqueta" para simular
producción.

**P: ¿Por qué Pydantic v2 validators y no JSON Schema o una capa de validación separada?**

R: Pydantic v2 ya viene con FastAPI, no añade dependencias. Los field_validators
están al lado del modelo de datos, no en un archivo separado. La validación ocurre
automáticamente antes de que el controlador toque los datos. Es el enfoque más
idiomático para FastAPI. JSON Schema requeriría un middleware separado.

---

## 4. Sobre operaciones y despliegue

**P: ¿Qué pasa si la API se cae? ¿Dejamos pasar todo el fraude?**

R: La API es stateless (el modelo está en memoria, no en BD). Si se cae:
1. **Short-term**: un proxy inverso (nginx) redirige a otra instancia.
   Docker compose con 2 réplicas da 99.9% uptime.
2. **Medium-term**: el proceso `uvicorn` se reinicia automáticamente con
   `--reload` o con `restart: unless-stopped` en Docker.
3. **Fallback**: si la API no responde en <500ms, el sistema puede optar por
   bloquear (conservador) o dejar pasar (permisivo). Es una decisión de negocio.

Actualmente no hay fallback implementado. Habría que decidir la política.

**P: ¿Cuánto tarda en arrancar la API?**

R: ~5-10 segundos. La carga del `.pkl` con joblib es el cuello de botella
(~200MB deserializados). Para arranques frecuentes (auto-scaling en Kubernetes),
se puede precargar el modelo en un sidecar o usar memoria compartida.
Para el caso actual (1-2 instancias), 10 segundos no es problema.

**P: ¿Y si el tráfico se dispara (Black Friday, Navidad)?**

R: El modelo infiere en ~50ms por transacción en CPU. Para 200K tx/día,
eso son ~3h CPU/día, asumible. Si el tráfico se multiplica por 10 (2M tx/día),
necesitaríamos escalar a más instancias. El modelo es stateless → escalado
horizontal directo. El cuello de botella sería la BD (PostgreSQL) si registramos
cada predicción.

Con 2M tx/día y 50ms cada una, son ~28h CPU/día → 2 instancias (~14h cada una)
o 4 instancias (~7h cada una). El coste es lineal.

**P: ¿Qué versión de Python usáis y por qué?**

R: Python 3.14 (última estable a fecha del proyecto). Las dependencias clave:
scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.2.0. Elegimos las más recientes
por mejoras de rendimiento en inferencia (scikit-learn 1.8 tiene optimizaciones
en KNNImputer). El dockerfile usa python:3.11-slim como base.

---

## 5. Sobre costes y ROI

**P: ¿Cuánto ha costado desarrollar esto?**

R: Estimación: 1 Data Scientist × 3 meses = ~30.000€ (coste empresa España).
El coste de infraestructura (cloud, Docker) es despreciable (<100€/mes).
El coste operativo principal son los analistas: 30.000€/año cada uno.

**Total año 1**: 30.000€ (desarrollo) + 120.000€ (4 analistas) = 150.000€.
**Fraude bloqueado estimado**: 7.000 tx/día × 300€ × 84.7% recall × 365 días ×
12.9% (revisado por analistas) + 71.8% (bloqueo automático sin revisión) = ~465M€/año
de fraude bloqueado (automático + revisado).

ROI: 465M€ / 150K€ = **3.100:1**. Cada euro invertido recupera ~3.100€.

**P: ¿No es más barato comprar una solución de fraude de un proveedor?**

R: Depende del volumen y la personalización. Las soluciones comerciales (Feedzai,
SAS, FICO) cuestan 100K-500K€/año + comisión por transacción. Suelen venir con
modelos pre-entrenados pero difíciles de personalizar para patrones específicos
de NovaPay. Nuestra solución es específica para nuestros datos, canales y equipo.
El coste de desarrollo (30K€) es recuperable en el primer mes si el modelo funciona.

Si NovaPay procesara 1M tx/mes, una solución comercial podría costar 200K€/año.
Nuestra solución cuesta 150K€/año (incluyendo 4 analistas). Y tenemos control
total sobre el modelo, las features y los thresholds.

**P: ¿Cuándo dejáis de usar datos sintéticos y usáis solo reales?**

R: Cuando tengamos ~50.000 transacciones reales etiquetadas por analistas.
Con 4 analistas revisando 1.000 tx/día y ~12.9% siendo fraude, obtenemos
~129 fraudes reales/día. En ~400 días (1 año+), tendríamos 50.000 fraudes reales.
Hasta entonces, los datos sintéticos son necesarios para aumentar el conjunto
de entrenamiento.

La estrategia es **híbrida**: dataset sintético (base) + datos reales etiquetados
(refuerzo). Progresivamente, el peso de los datos reales aumenta y el sintético
disminuye. Para v4, estimamos un 70% sintético / 30% real.

---

## 6. Sobre ética, regulatorio y cumplimiento

**P: ¿El modelo puede ser discriminatorio? ¿Cómo lo evitáis?**

R: Es una preocupación legítima. Hoy no tenemos variables demográficas sensibles
explícitas (raza, género, religión). Sí tenemos edad y país, que pueden correlacionar
con grupos protegidos.

**Lo que hacemos hoy**:
- No incluimos género, raza, ni datos biométricos
- Las features geográficas (país, región) se usan como señales de fraude
  (cross-border, no discriminación por nacionalidad)
- El threshold se ajusta por canal, no por perfil demográfico

**Lo que deberíamos hacer** (no implementado):
1. Análisis de equidad (fairness) del modelo: comparar precisión/recall por
   segmentos de edad, país y tipo_cliente
2. Si se detectan disparidades, ajustar thresholds por segmento para equilibrar
3. Auditoría externa antes de desplegar en producción real

**P: ¿Cumplimos con GDPR? ¿Qué datos personales guardamos?**

R: El sistema actual trabaja con datos sintéticos, no reales. Los datos sintéticos
no contienen información personal real. En producción real:
- Se guardan `id_transaccion`, `id_cliente`, `id_cuenta`, `id_tarjeta`
  (identificadores, no datos personales)
- `fecha_hora`, `importe_transaccion` (datos transaccionales)
- `direccion_ip_origen` (dato personal según GDPR — IP es PI)
- `geolocalizacion` (dato personal)

Habría que evaluar si `direccion_ip_origen` y `geolocalizacion` son necesarias
para el modelo. Si no aportan significativamente al PR-AUC, deberían eliminarse
para minimizar datos personales procesados (privacidad por diseño).

**P: ¿El modelo es interpretable? ¿Podemos explicar por qué se bloqueó una transacción?**

R: Sí, por dos vías:
1. **Output de la API**: los 10 campos incluyen `es_transfronteriza`,
   `intensidad_tx`, `flujo_neto_30d` que son features interpretables.
   El analista puede ver por qué se marcó.
2. **SHAP values**: podemos calcular la contribución de cada feature a la decisión.
   No está integrado en la API (aumentaría latencia) pero se puede calcular offline
   para casos disputados.

Para un cliente, la explicación sería: "Su transacción fue bloqueada porque se
realizó desde el extranjero en un dispositivo no reconocido, con un importe muy
superior a su media mensual." No se dice "el algoritmo lo decidió."

**P: ¿Qué pasa si un cliente disputa un bloqueo y tenía razón?**

R: El analista revisa, confirma que es falso positivo y actualiza `target_final=FALSE`.
Ese caso se incorpora al reentrenamiento. El modelo aprende de su error.
Además, se pueden crear reglas de negocio para que patrones similares no vuelvan a
generar falsos positivos (por ejemplo, "si el cliente viaja a X país, no bloquear
cross-border desde allí durante 7 días").

---

## 7. Sobre debilidades no mencionadas (autocrítica)

**P: ¿Cuál es la mayor debilidad del proyecto hoy?**

R: **La validación con datos reales.** El modelo solo se ha probado con datos
sintéticos. Aunque los datos sintéticos son realistas, no hay garantía de que
el rendimiento se sostenga en producción real. Hemos mitigado esto con:
- Inyección de señal basada en literatura de fraude real
- Monitoreo de drift desde el día 1
- Pipeline de reentrenamiento con feedback de analistas

Pero el elefante en la habitación es que el PR-AUC 0.90 y recall 84.7% son
**mediciones sobre datos sintéticos**. El rendimiento real podría ser inferior.
Es la principal fuente de incertidumbre del proyecto.

**P: ¿Qué es lo que menos os gusta del modelo actual?**

R: Tres cosas:
1. **ECE 0.0616**: la calibración es mejorable. El analista no puede confiar
   ciegamente en prob_fraud como probabilidad real.
2. **Solo 3 perfiles de generación de datos sintéticos**: los patrones de fraude
   son más diversos que "mixto/sospechoso/fraude". Necesitamos más variedad
   para cubrir bordes de decisión.
3. **Sin A/B testing**: desplegamos v3 directamente, sin comparar contra v2
   en producción con tráfico real dividido. Nos faltan herramientas para
   hacer A/B testing de modelos (nginx con split de tráfico).

**P: ¿Qué métrica os preocupa más si esto sale a producción?**

R: La **estabilidad del threshold por canal**. Optimizamos el threshold F2 sobre
validación con datos sintéticos. En producción, las distribuciones pueden ser
distintas y el threshold óptimo puede desplazarse. Si el threshold es demasiado
bajo → muchos falsos positivos. Si es demasiado alto → se escapan fraudes.

Por eso el health endpoint reporta los thresholds y recomendamos monitorizar
semanalmente la tasa de fraude detectada por canal. Si `tarjeta` pasa de tener
un 3.5% de flagging a un 8%, algo cambió.

**P: ¿Qué pasaría si mañana os pidiera poner esto en producción real?**

R: Lo haríamos con estas condiciones:
1. **Primeros 7 días en shadow mode**: el modelo predice pero no bloquea.
   Comparamos sus decisiones con las del sistema actual (si existe).
2. **Días 8-14 en modo alerta**: el modelo marca fraude pero el bloqueo lo
   decide un analista. Medimos precisión real.
3. **Día 15**: si precisión > 70%, activamos bloqueo automático con el top 0.1%
   (precisión 100% esperada) y alertas para el resto.
4. **Monitoreo continuo**: PSI > 0.1 o KS-test p < 0.05 → alerta.
   Si la precisión cae por debajo de 60% → volver a modo alerta.

No lo desplegaríamos sin este plan de rollout graduado.

**P: ¿Qué es lo primero que haríais si os dieran 3 meses más?**

R: 1) Recalibración isotónica (1 semana). 2) SHAP analysis para feature selection
(2 semanas). 3) Integración de rondas reales de Ciberseguridad en el dataset
de entrenamiento (4 semanas). 4) A/B testing framework con nginx (2 semanas).
5) Dos rondas de hard negative mining para mejorar recall@k (3 semanas).
El resto del tiempo: testeo y validación.

---

## 8. Preguntas rápidas

**P: ¿Cuántas features tiene el modelo?** 69 (v4).

**P: ¿Cuánto pesa el .pkl?** ~200MB.

**P: ¿Cada cuánto se reentrena?** Cuando hay una ronda de Ciberseguridad o cuando
el drift detection lo indica. Estimado: cada 2-4 semanas.

**P: ¿Qué canal tiene más fraude?** Depende del perfil, pero en v3 bizum tiene
menos histórico y por eso su threshold es más conservador.

**P: ¿Se puede ejecutar en una Raspberry Pi?** No, el modelo requiere ~200MB RAM
y ~50ms de CPU. Una Raspberry Pi 5 podría ejecutarlo pero a ~500ms por inferencia.
No es práctico.

**P: ¿Cuánto tiempo de vida tiene el modelo actual antes de degradarse?** Depende
del cambio en patrones de fraude. Con monitoreo de drift, esperamos detectar
degradación antes de que pierda >5% de PR-AUC. Estimación conservadora: 2-4 meses.

**P: ¿Por qué no usáis CatBoost?** Porque no estaba instalado en el entorno inicial.
Los notebooks lo manejan con try/except. Si se añade al requirements, se puede
incorporar fácilmente. CatBoost maneja categóricas nativamente y podría simplificar
el FeatureEngineer.

**P: ¿Qué pasa con los valores nulos?** KNNImputer n=5. Los nulos vienen de
divisiones por cero en `_safe_ratio`. KNN estima el valor basándose en los 5
vecinos más cercanos, preservando correlaciones.

**P: ¿Habéis probado con datos reales aunque sea un piloto pequeño?** No hoy.
Es la prioridad #1 si el proyecto sigue adelante.

**P: ¿Cuánto tarda el entrenamiento completo de v3?** ~2 minutos en CPU
(100K registros, 69 features, LGB+XGB con validación y búsqueda de threshold).

**P: ¿Cuántas transacciones puede procesar la API por segundo?** ~20 tx/s en CPU
single-thread. Con 200K tx/día (2,3 tx/s), hay margen de ~10×.
