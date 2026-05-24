"""
Ejemplo de inferencia con modelos guardados.

Uso:
    python scripts/inference_example.py

Los .pkl contienen:
    fe        -> FeatureEngineer (fitted v3)
    scaler    -> StandardScaler (fitted)
    imputer   -> KNNImputer (fitted, n_neighbors=5)
    lgb_model -> LightGBM entrenado
    xgb_model -> XGBoost entrenado
    best_w    -> peso del ensemble (LightGBM)
    best_t    -> threshold F2
    num_feats -> lista de columnas numéricas
    metadata  -> métricas de entrenamiento
"""

import sys, joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. Cargar modelo
# ============================================================
# Ejecutar desde la raíz del proyecto
sys.path.append(str(Path.cwd()))
from scripts.feature_engineering import FeatureEngineer

model_path = Path.cwd() / 'scripts' / 'saved_models' / 'modelo_08_v2.pkl'
obj = joblib.load(model_path)

fe = obj['fe']
scaler = obj['scaler']
imputer = obj['imputer']
lgb_model = obj['lgb_model']
xgb_model = obj['xgb_model']
best_w = obj['best_w']
best_t = obj['best_t']
num_feats = obj['num_feats']

print(f'Modelo: {obj["metadata"]["label"]}')
print(f'  Ensemble: w(LGB)={best_w:.3f} + w(XGB)={1-best_w:.3f}')
print(f'  Threshold F2: {best_t:.4f}')
print(f'  Features: {len(num_feats)} numéricas')

# ============================================================
# 2. Cargar datos nuevos (mismas columnas que entrenamiento)
# ============================================================
df = pd.read_csv(Path.cwd() / 'data' / 'dataset_fraude_mejorado.csv')
y_true = df['IS_FRAUD'].values

# ============================================================
# 3. Pipeline de inferencia
# ============================================================
X = fe.transform(df)
X = X.drop(columns=['IS_FRAUD'], errors='ignore')

X_s = X.copy()
X_s[num_feats] = scaler.transform(X[num_feats])
X_s[num_feats] = imputer.transform(X_s[num_feats])

# ============================================================
# 4. Ensemble + Threshold
# ============================================================
lgb_prob = lgb_model.predict_proba(X_s)[:, 1]
xgb_prob = xgb_model.predict_proba(X_s)[:, 1]
y_prob = best_w * lgb_prob + (1 - best_w) * xgb_prob
y_pred = (y_prob >= best_t).astype(int)

# ============================================================
# 5. Evaluación
# ============================================================
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
print()
print(classification_report(y_true, y_pred, digits=4))
print(f'PR-AUC:  {average_precision_score(y_true, y_prob):.4f}')
print(f'AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}')
print(f'Fraudes detectados: {y_pred.sum()} / {y_true.sum()} ({y_pred[y_true==1].sum()})')
print(f'Alertas totales: {y_pred.sum()}')
