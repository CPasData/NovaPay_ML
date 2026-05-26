import warnings, sys, joblib, json
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

base = Path(__file__).resolve().parent
sys.path.append(str(base))
from feature_engineering import FeatureEngineer

# Leer el source completo para embeberlo en el .pkl
FE_SOURCE = (base / 'feature_engineering.py').read_text(encoding='utf-8')

models_dir = base / 'saved_models'
models_dir.mkdir(exist_ok=True)

# Copiar feature_engineering.py junto a los .pkl para carga autónoma
import shutil
shutil.copy2(str(base / 'feature_engineering.py'), str(models_dir / 'feature_engineering.py'))

np.random.seed(42)


# CONFIG

datasets = [
    {
        'name': 'v1',
        'label': 'v1 (original)',
        'data_path': base.parent / 'data' / 'dataset_fraude.csv',
        'savename': 'modelo_07_v1.pkl',
        'xgb_params': {
            'learning_rate': 0.05, 'max_depth': 3,
            'min_child_weight': 1, 'n_estimators': 200,
            'reg_lambda': 0, 'subsample': 1.0,
        },
    },
    {
        'name': 'v2',
        'label': 'v2 (mejorado)',
        'data_path': base.parent / 'data' / 'dataset_fraude_mejorado.csv',
        'savename': 'modelo_08_v2.pkl',
        'xgb_params': {
            'learning_rate': 0.05, 'max_depth': 3,
            'min_child_weight': 1, 'n_estimators': 200,
            'reg_lambda': 0, 'subsample': 1.0,
        },
    },
]

def train_pipeline(cfg):
    print(f'\n{"="*60}')
    print(f'Modelo: {cfg["label"]}')
    print(f'{"="*60}')

    # 1. Carga
    df = pd.read_csv(cfg['data_path'])
    other = [c for c in ['IS_FRAUD', 'IMPACTO_FRAUDE'] if c != 'IS_FRAUD']
    df = df.drop(columns=other, errors='ignore')
    print(f'  Filas: {len(df):,}  Fraude: {df["IS_FRAUD"].sum():,} ({df["IS_FRAUD"].mean()*100:.2f}%)')

    # 2. Feature Engineering v3
    fe = FeatureEngineer(encode_target='IS_FRAUD', random_state=42)
    X = fe.fit_transform(df)
    y = X.pop('IS_FRAUD').values
    print(f'  Features: {X.shape[1]}')

    # 3. Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f'  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}')

    # 4. Scale + Impute
    num_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_train_s[num_feats] = scaler.fit_transform(X_train[num_feats])
    X_val_s = X_val.copy()
    X_val_s[num_feats] = scaler.transform(X_val[num_feats])
    X_test_s = X_test.copy()
    X_test_s[num_feats] = scaler.transform(X_test[num_feats])

    imputer = KNNImputer(n_neighbors=5)
    X_train_s[num_feats] = imputer.fit_transform(X_train_s[num_feats])
    X_val_s[num_feats] = imputer.transform(X_val_s[num_feats])
    X_test_s[num_feats] = imputer.transform(X_test_s[num_feats])

    # 5. XGBoost
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_params = cfg['xgb_params'].copy()
    xgb_params['scale_pos_weight'] = scale_pos
    xgb_model = xgb.XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss', **xgb_params)
    xgb_model.fit(X_train_s, y_train)

    yprob_xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
    yprob_xgb_test = xgb_model.predict_proba(X_test_s)[:, 1]
    prauc_xgb = average_precision_score(y_val, yprob_xgb_val)
    auc_xgb = roc_auc_score(y_val, yprob_xgb_val)
    print(f'  XGBoost Val:   PR-AUC={prauc_xgb:.4f}  AUC-ROC={auc_xgb:.4f}')

    # 6. Calibración (opcional)
    cal_xgb = CalibratedClassifierCV(xgb_model, cv=3, method='sigmoid')
    cal_xgb.fit(X_train_s, y_train)

    r = average_precision_score(y_val, xgb_model.predict_proba(X_val_s)[:, 1])
    c = average_precision_score(y_val, cal_xgb.predict_proba(X_val_s)[:, 1])
    use_cal = (c - r) > 0.01
    print(f'  XGBoost: Raw={r:.4f}  Cal={c:.4f}  Diff={c-r:+.4f}')
    if use_cal:
        print('  >> Usando probabilidades calibradas')
        yprob_val = cal_xgb.predict_proba(X_val_s)[:, 1]
        yprob_test = cal_xgb.predict_proba(X_test_s)[:, 1]
    else:
        print('  >> Probabilidades raw (calibraci\u00f3n no mejora)')
        yprob_val = xgb_model.predict_proba(X_val_s)[:, 1]
        yprob_test = xgb_model.predict_proba(X_test_s)[:, 1]

    # 7. F2 threshold selection on validation
    thrs = np.linspace(0.01, 0.99, 500)
    target_precision = 0.60
    best_t_60, best_rec_60 = None, -1
    results = []
    for t in thrs:
        yt = (yprob_val >= t).astype(int)
        p = precision_score(y_val, yt, zero_division=0)
        r = recall_score(y_val, yt, zero_division=0)
        results.append({'threshold': t, 'precision': p, 'recall': r})
        if p >= target_precision and r > best_rec_60:
            best_t_60, best_rec_60 = t, r

    results_df = pd.DataFrame(results)
    results_df['f2'] = (5 * results_df['precision'] * results_df['recall']) / \
                        (4 * results_df['precision'] + results_df['recall'] + 1e-10)
    best_f2_row = results_df.loc[results_df['f2'].idxmax()]
    best_t = best_f2_row['threshold']
    best_prec = best_f2_row['precision']
    best_rec = best_f2_row['recall']

    print(f'  Threshold F2: t={best_t:.4f}  Prec={best_prec:.4f}  Rec={best_rec:.4f}')
    if best_rec_60 > 0:
        print(f'  Precision>={target_precision:.0%}: t={best_t_60:.4f}  Rec={best_rec_60:.4f}')

    # 8. Test evaluation
    yp_test = (yprob_test >= best_t).astype(int)
    test_prauc = average_precision_score(y_test, yprob_test)
    test_auc = roc_auc_score(y_test, yprob_test)
    test_prec = precision_score(y_test, yp_test, zero_division=0)
    test_rec = recall_score(y_test, yp_test, zero_division=0)
    test_f1 = (2 * test_prec * test_rec) / (test_prec + test_rec + 1e-10)
    print(f'  Test: PR-AUC={test_prauc:.4f}  AUC-ROC={test_auc:.4f}  '
          f'Prec={test_prec:.4f}  Rec={test_rec:.4f}  F1={test_f1:.4f}')

    # 9. Build artifact
    artifact = {
        '_fe_source': FE_SOURCE,
        'fe': fe,
        'scaler': scaler,
        'imputer': imputer,
        'xgb_model': xgb_model,
        'best_t': best_t,
        'best_prec': best_prec,
        'best_rec': best_rec,
        'num_feats': num_feats,
        'metadata': {
            'dataset': cfg['name'],
            'label': cfg['label'],
            'fecha': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'n_features': X.shape[1],
            'n_train': X_train.shape[0],
            'n_val': X_val.shape[0],
            'n_test': X_test.shape[0],
            'fraud_rate': float(y.mean()),
            'model': 'XGBoost',
            'val_prauc': float(average_precision_score(y_val, yprob_val)),
            'val_auc': float(roc_auc_score(y_val, yprob_val)),
            'test_prauc': float(test_prauc),
            'test_auc': float(test_auc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'f2_threshold': float(best_t),
            'calibration_used': use_cal,
        },
    }

    path = models_dir / cfg['savename']
    joblib.dump(artifact, path)
    print(f'  Guardado: {path}')
    return artifact


# TRAIN BOTH

print('=== REGENERACI\u00d3N DE MODELOS ===')
print(f'Feature Engineering: v3 (67 features)')
print(f'Pipeline: FE -> Scale -> KNNImputer -> XGBoost -> F2 thr')
print()

results = {}
for cfg in datasets:
    results[cfg['name']] = train_pipeline(cfg)

# Summary
print(f'\n{"="*60}')
print('RESUMEN')
print(f'{"="*60}')
print(f'{"Dataset":15s} {"PR-AUC":>8s} {"AUC-ROC":>8s} {"Prec":>6s} {"Recall":>7s} {"F1":>6s} {"Thr":>6s}')
print(f'{ "-"*15:15s} {"-"*8:>8s} {"-"*8:>8s} {"-"*6:>6s} {"-"*7:>7s} {"-"*6:>6s} {"-"*6:>6s}')
for name in ['v1', 'v2']:
    r = results[name]
    m = r['metadata']
    print(f'{m["label"]:15s} {m["test_prauc"]:>8.4f} {m["test_auc"]:>8.4f} '
          f'{m["test_precision"]:>6.1%} {m["test_recall"]:>6.1%} '
          f'{m["test_f1"]:>6.4f} {m["f2_threshold"]:>6.4f}')

print()
print('OK')
