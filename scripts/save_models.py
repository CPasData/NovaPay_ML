"""
LEGACY - reemplazado por regenerate_models.py
Los modelos guardados con este script NO incluyen XGBoost, ensemble weights,
KNNImputer, ni threshold F2. Solo se conserva como referencia.
"""
import warnings, sys, joblib
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import FeatureEngineer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

base = Path(__file__).resolve().parent
models_dir = base / 'saved_models'
models_dir.mkdir(exist_ok=True)

# === MODELO 07: v1 original ===
print('=== Modelo 07 (v1 original) ===')
df = pd.read_csv(base.parent / 'data' / 'dataset_fraude.csv')
df = df.drop(columns=['IMPACTO_FRAUDE'], errors='ignore')
print(f'  Filas: {len(df)}  Fraude: {df["IS_FRAUD"].sum()} ({df["IS_FRAUD"].mean()*100:.2f}%)')

fe = FeatureEngineer(encode_target='IS_FRAUD', random_state=42)
X = fe.fit_transform(df)
y = X.pop('IS_FRAUD').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = lgb.LGBMClassifier(
    learning_rate=0.05, min_child_samples=20, n_estimators=200,
    num_leaves=15, reg_lambda=10,
    scale_pos_weight=float(6.66), subsample=0.8,
    random_state=42, verbose=-1
)
model.fit(X_train_s, y_train)
yprob = model.predict_proba(X_test_s)[:, 1]
print(f'  PR-AUC test: {average_precision_score(y_test, yprob):.4f}  AUC-ROC: {roc_auc_score(y_test, yprob):.4f}')

joblib.dump({'model': model, 'scaler': scaler, 'fe': fe}, models_dir / 'modelo_07_v1.pkl')
print(f'  Guardado: {models_dir / "modelo_07_v1.pkl"}')

# === MODELO 08: v2 mejorado ===
print()
print('=== Modelo 08 (v2 mejorado) ===')
df2 = pd.read_csv(base.parent / 'data' / 'dataset_fraude_mejorado.csv')
df2 = df2.drop(columns=['IMPACTO_FRAUDE'], errors='ignore')
print(f'  Filas: {len(df2)}  Fraude: {df2["IS_FRAUD"].sum()} ({df2["IS_FRAUD"].mean()*100:.2f}%)')

fe2 = FeatureEngineer(encode_target='IS_FRAUD', random_state=42)
X2 = fe2.fit_transform(df2)
y2 = X2.pop('IS_FRAUD').values

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
scaler2 = StandardScaler()
X_train2_s = scaler2.fit_transform(X_train2)
X_test2_s = scaler2.transform(X_test2)

model2 = lgb.LGBMClassifier(
    learning_rate=0.1, min_child_samples=100, n_estimators=200,
    num_leaves=15, reg_lambda=10,
    scale_pos_weight=float(6.78), subsample=0.8,
    random_state=42, verbose=-1
)
model2.fit(X_train2_s, y_train2)
yprob2 = model2.predict_proba(X_test2_s)[:, 1]
print(f'  PR-AUC test: {average_precision_score(y_test2, yprob2):.4f}  AUC-ROC: {roc_auc_score(y_test2, yprob2):.4f}')

joblib.dump({'model': model2, 'scaler': scaler2, 'fe': fe2}, models_dir / 'modelo_08_v2.pkl')
print(f'  Guardado: {models_dir / "modelo_08_v2.pkl"}')

print()
print('OK')
