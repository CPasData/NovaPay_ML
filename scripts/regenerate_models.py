import warnings, sys, joblib, json
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

base = Path(__file__).resolve().parent
sys.path.insert(0, str(base))
from feature_engineering import FeatureEngineer

# Leer el source completo para embeberlo en el .pkl
FE_SOURCE = (base / 'feature_engineering.py').read_text(encoding='utf-8')

models_dir = base.parent / 'model'
models_dir.mkdir(exist_ok=True)

np.random.seed(42)


# CONFIG

datasets = [
    {
        'name': 'v1',
        'label': 'v1 (original)',
        'data_path': base.parent / 'data' / 'dataset_fraude.csv',
        'savename': 'modelo_07_v1.pkl',
        'lgb_params': {
            'learning_rate': 0.05, 'min_child_samples': 20,
            'n_estimators': 200, 'num_leaves': 15,
            'reg_lambda': 10, 'subsample': 0.8,
        },
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
        'lgb_params': {
            'learning_rate': 0.1, 'min_child_samples': 100,
            'n_estimators': 200, 'num_leaves': 15,
            'reg_lambda': 10, 'subsample': 0.8,
        },
        'xgb_params': {
            'learning_rate': 0.05, 'max_depth': 3,
            'min_child_weight': 1, 'n_estimators': 200,
            'reg_lambda': 0, 'subsample': 1.0,
        },
    },
    {
        'name': 'v3',
        'label': 'v3 (3% fraude)',
        'data_path': base.parent / 'data' / 'dataset_fraude_v3.csv',
        'savename': 'modelo_11_v3.pkl',
        'lgb_params': {
            'learning_rate': 0.1, 'min_child_samples': 100,
            'n_estimators': 200, 'num_leaves': 15,
            'reg_lambda': 10, 'subsample': 0.8,
        },
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

    imputer = SimpleImputer(strategy='median')
    X_train_s[num_feats] = imputer.fit_transform(X_train_s[num_feats])
    X_val_s[num_feats] = imputer.transform(X_val_s[num_feats])
    X_test_s[num_feats] = imputer.transform(X_test_s[num_feats])

    # 5. LightGBM
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    lgb_params = cfg['lgb_params'].copy()
    lgb_params['scale_pos_weight'] = scale_pos
    lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1, **lgb_params)
    lgb_model.fit(X_train_s, y_train)

    yprob_lgb_val = lgb_model.predict_proba(X_val_s)[:, 1]
    yprob_lgb_test = lgb_model.predict_proba(X_test_s)[:, 1]
    prauc_lgb = average_precision_score(y_val, yprob_lgb_val)
    auc_lgb = roc_auc_score(y_val, yprob_lgb_val)
    print(f'  LightGBM Val: PR-AUC={prauc_lgb:.4f}  AUC-ROC={auc_lgb:.4f}')

    # 6. XGBoost
    xgb_params = cfg['xgb_params'].copy()
    xgb_params['scale_pos_weight'] = scale_pos
    xgb_model = xgb.XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss', **xgb_params)
    xgb_model.fit(X_train_s, y_train)

    yprob_xgb_val = xgb_model.predict_proba(X_val_s)[:, 1]
    yprob_xgb_test = xgb_model.predict_proba(X_test_s)[:, 1]
    prauc_xgb = average_precision_score(y_val, yprob_xgb_val)
    auc_xgb = roc_auc_score(y_val, yprob_xgb_val)
    print(f'  XGBoost Val:   PR-AUC={prauc_xgb:.4f}  AUC-ROC={auc_xgb:.4f}')

    # 7. Calibración (opcional)
    cal_lgb = CalibratedClassifierCV(lgb_model, cv=3, method='sigmoid')
    cal_lgb.fit(X_train_s, y_train)
    cal_xgb = CalibratedClassifierCV(xgb_model, cv=3, method='sigmoid')
    cal_xgb.fit(X_train_s, y_train)

    use_cal = False
    for raw, cal, name in [(lgb_model, cal_lgb, 'LightGBM'), (xgb_model, cal_xgb, 'XGBoost')]:
        r = average_precision_score(y_val, raw.predict_proba(X_val_s)[:, 1])
        c = average_precision_score(y_val, cal.predict_proba(X_val_s)[:, 1])
        diff = c - r
        if diff > 0.01:
            use_cal = True
        print(f'  {name}: Raw={r:.4f}  Cal={c:.4f}  Diff={diff:+.4f}')

    if use_cal:
        print('  >> Usando probabilidades calibradas')
        yprob_lgb_val = cal_lgb.predict_proba(X_val_s)[:, 1]
        yprob_xgb_val = cal_xgb.predict_proba(X_val_s)[:, 1]
        yprob_lgb_test = cal_lgb.predict_proba(X_test_s)[:, 1]
        yprob_xgb_test = cal_xgb.predict_proba(X_test_s)[:, 1]
    else:
        print('  >> Probabilidades raw (calibraci\u00f3n no mejora)')

    # 8. Ensemble weight optimization
    weights = np.linspace(0, 1, 101)
    best_w, best_prauc = 0.5, 0
    for w in weights:
        yprob_ens = w * yprob_lgb_val + (1 - w) * yprob_xgb_val
        prauc = average_precision_score(y_val, yprob_ens)
        if prauc > best_prauc:
            best_prauc, best_w = prauc, w

    yprob_val = best_w * yprob_lgb_val + (1 - best_w) * yprob_xgb_val
    yprob_test = best_w * yprob_lgb_test + (1 - best_w) * yprob_xgb_test
    print(f'  Ensemble: w={best_w:.3f} (LGB) + {1-best_w:.3f} (XGB)  PR-AUC={best_prauc:.4f}')

    # 9. F2 threshold selection on validation
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

    # 10. Test evaluation
    yp_test = (yprob_test >= best_t).astype(int)
    test_prauc = average_precision_score(y_test, yprob_test)
    test_auc = roc_auc_score(y_test, yprob_test)
    test_prec = precision_score(y_test, yp_test, zero_division=0)
    test_rec = recall_score(y_test, yp_test, zero_division=0)
    test_f1 = (2 * test_prec * test_rec) / (test_prec + test_rec + 1e-10)
    print(f'  Test: PR-AUC={test_prauc:.4f}  AUC-ROC={test_auc:.4f}  '
          f'Prec={test_prec:.4f}  Rec={test_rec:.4f}  F1={test_f1:.4f}')

    # 11. Brier Score y ECE (Expected Calibration Error)
    brier = brier_score_loss(y_test, yprob_test)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ece = 0.0
    for i in range(n_bins):
        in_bin = (yprob_test >= bin_edges[i]) & (yprob_test < bin_edges[i + 1])
        if in_bin.sum() > 0:
            acc = y_test[in_bin].mean()
            conf = yprob_test[in_bin].mean()
            ece += (in_bin.sum() / len(y_test)) * abs(acc - conf)
    print(f'  Brier={brier:.4f}  ECE={ece:.4f}')

    # 12. Per-channel threshold optimization (solo para v3 con tipo_transaccion)
    canales = ['tarjeta', 'transferencia', 'bizum']
    per_channel_thresholds = {}
    if cfg['name'] == 'v3' and 'tipo_transaccion' in df.columns:
        df_val = df.loc[X_val.index]
        for canal in canales:
            mask = df_val['tipo_transaccion'].values == canal
            if mask.sum() < 10:
                continue
            yc = y_val[mask]
            pc = yprob_val[mask]
            thrs_c = np.linspace(0.01, 0.99, 200)
            best_f2_c, best_t_c = -1, best_t
            for t in thrs_c:
                yc_pred = (pc >= t).astype(int)
                n_pos = yc_pred.sum()
                if n_pos == 0:
                    continue
                p_c = precision_score(yc, yc_pred, zero_division=0)
                r_c = recall_score(yc, yc_pred, zero_division=0)
                f2_c = (5 * p_c * r_c) / (4 * p_c + r_c + 1e-10)
                if f2_c > best_f2_c:
                    best_f2_c, best_t_c = f2_c, t
            per_channel_thresholds[canal] = float(best_t_c)
        print(f'  Thresholds por canal: {per_channel_thresholds}')

    # 13. Recall@k (simula produccion: 200 alertas / 200k tx = 0.1%)
    k_pct = 0.001  # top 0.1%
    k = max(1, int(len(y_test) * k_pct))
    top_k_idx = np.argsort(yprob_test)[-k:]
    total_frauds_test = int(y_test.sum())
    frauds_in_top_k = int(y_test[top_k_idx].sum())
    recall_at_k = frauds_in_top_k / total_frauds_test if total_frauds_test > 0 else 0
    alert_precision = frauds_in_top_k / k
    print(f'  Recall@k (k={k}, {k_pct*100:.1f}%): {recall_at_k:.4f}  '
          f'Fraudes capturados={frauds_in_top_k}/{total_frauds_test}  '
          f'Precision en alertas={alert_precision:.4f}')
    if cfg['name'] == 'v3':
        print(f'  >> Equivale a {frauds_in_top_k} fraudes capturados de {total_frauds_test} en ~200 alertas diarias')

    # 14. Build artifact
    artifact = {
        '_fe_source': FE_SOURCE,
        'fe': fe,
        'scaler': scaler,
        'imputer': imputer,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'best_w': best_w,
        'best_t': best_t,
        'best_prec': best_prec,
        'best_rec': best_rec,
        'num_feats': num_feats,
        'metadata': {
            'dataset': cfg['name'],
            'label': cfg['label'],
            'fecha': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'n_features': X.shape[1],
            'n_train': X_train.shape[0],
            'n_val': X_val.shape[0],
            'n_test': X_test.shape[0],
            'fraud_rate': float(y.mean()),
            'lightgbm_val_prauc': float(prauc_lgb),
            'lightgbm_val_auc': float(auc_lgb),
            'xgboost_val_prauc': float(prauc_xgb),
            'xgboost_val_auc': float(auc_xgb),
            'ensemble_val_prauc': float(best_prauc),
            'ensemble_test_prauc': float(test_prauc),
            'ensemble_test_auc': float(test_auc),
            'ensemble_test_precision': float(test_prec),
            'ensemble_test_recall': float(test_rec),
            'ensemble_test_f1': float(test_f1),
            'best_w': float(best_w),
            'f2_threshold': float(best_t),
            'calibration_used': use_cal,
            'brier_score': float(brier),
            'expected_calibration_error': float(ece),
            'recall_at_k': float(recall_at_k),
            'recall_at_k_pct': k_pct,
            'alert_precision': float(alert_precision),
            'frauds_in_top_k': int(frauds_in_top_k),
            'total_frauds_test': int(total_frauds_test),
        },
    }

    if per_channel_thresholds:
        artifact['per_channel_thresholds'] = per_channel_thresholds

    path = models_dir / cfg['savename']
    joblib.dump(artifact, path)
    print(f'  Guardado: {path}')
    return artifact


# TRAIN BOTH

print('=== REGENERACI\u00d3N DE MODELOS ===')
print(f'Feature Engineering: v4 (z-score, sin high_ratio_redondeado)')
print(f'Pipeline: FE -> Scale -> SimpleImputer(median) -> LGB/XGB -> Ensemble -> F2 thr | per-channel thr | recall@k')
print()

results = {}
for cfg in datasets:
    results[cfg['name']] = train_pipeline(cfg)

# Summary
print(f'\n{"="*60}')
print('RESUMEN')
print(f'{"="*60}')
print(f'{"Dataset":15s} {"PR-AUC":>8s} {"AUC-ROC":>8s} {"Prec":>6s} {"Recall":>7s} {"F1":>6s} {"Thr":>6s} {"w(LGB)":>7s} {"Rec@k":>7s} {"Brier":>7s} {"ECE":>7s}')
print(f'{ "-"*15:15s} {"-"*8:>8s} {"-"*8:>8s} {"-"*6:>6s} {"-"*7:>7s} {"-"*6:>6s} {"-"*6:>6s} {"-"*7:>7s} {"-"*7:>7s} {"-"*7:>7s} {"-"*7:>7s}')
for name in ['v1', 'v2', 'v3']:
    r = results[name]
    m = r['metadata']
    print(f'{m["label"]:15s} {m["ensemble_test_prauc"]:>8.4f} {m["ensemble_test_auc"]:>8.4f} '
          f'{m["ensemble_test_precision"]:>6.1%} {m["ensemble_test_recall"]:>6.1%} '
          f'{m["ensemble_test_f1"]:>6.4f} {m["f2_threshold"]:>6.4f} {m["best_w"]:>7.3f} '
          f'{m.get("recall_at_k", 0):>7.4f} {m.get("brier_score", 0):>7.4f} {m.get("expected_calibration_error", 0):>7.4f}')

