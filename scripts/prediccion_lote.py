"""
Genera un CSV con los datos originales más las predicciones del modelo.

Lee un CSV de transacciones (con o sin IS_FRAUD), ejecuta el pipeline completo
y guarda un nuevo CSV con las columnas originales + probabilidad, predicción binaria
e impacto estimado.

Uso:
    python scripts/prediccion_lote.py --input data/dataset_fraude_mejorado.csv --output data/resultado.csv
    python scripts/prediccion_lote.py --input data/muestra_sin_etiqueta.csv --output data/predicciones.csv --modelo v2
"""

import sys, joblib, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import FeatureEngineer


def calcular_impacto(is_fraud, importe):
    if is_fraud == 0:
        return 0
    elif importe < 500:
        return 1
    elif importe < 2000:
        return 2
    else:
        return 3


def main():
    parser = argparse.ArgumentParser(description='Batch inference: datos + predicciones en CSV')
    parser.add_argument('--input', required=True, help='CSV de entrada con transacciones')
    parser.add_argument('--output', required=True, help='CSV de salida con predicciones')
    parser.add_argument('--modelo', choices=['v1', 'v2'], default='v2',
                        help='Modelo a usar (v1=original, v2=mejorado)')
    parser.add_argument('--no-impacto', action='store_true',
                        help='No incluir columna impacto_fraude')
    args = parser.parse_args()

    model_name = 'modelo_07_v1' if args.modelo == 'v1' else 'modelo_08_v2'
    model_path = Path(__file__).resolve().parent.parent / 'model' / f'{model_name}.pkl'
    if not model_path.exists():
        print(f'ERROR: No se encuentra {model_path}')
        sys.exit(1)

    print(f'Cargando modelo: {model_path.name}')
    obj = joblib.load(model_path)

    fe = obj['fe']
    scaler = obj['scaler']
    imputer = obj['imputer']
    lgb_model = obj['lgb_model']
    xgb_model = obj['xgb_model']
    best_w = obj['best_w']
    best_t = obj['best_t']
    num_feats = obj['num_feats']

    print(f'Ensemble: w(LGB)={best_w:.3f} + w(XGB)={1-best_w:.3f}')
    print(f'Threshold F2: {best_t:.4f}')
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'ERROR: No se encuentra {input_path}')
        sys.exit(1)

    print(f'Leyendo: {input_path}')
    df = pd.read_csv(input_path)
    n_total = len(df)
    print(f'Registros: {n_total:,}')
    tiene_label = 'IS_FRAUD' in df.columns
    if tiene_label:
        n_fraude = df['IS_FRAUD'].sum()
        print(f'IS_FRAUD presente: {n_fraude} fraudes ({n_fraude/n_total*100:.2f}%)')
    else:
        print('IS_FRAUD ausente — solo predicción')

    print('Ejecutando pipeline de inferencia...')
    X = fe.transform(df)

    # Anadir columnas de feature engineering al output
    fe_cols = [c for c in X.columns if c not in df.columns and c != 'IS_FRAUD']
    for col in fe_cols:
        df[col] = X[col].values

    X = X.drop(columns=['IS_FRAUD'], errors='ignore')

    X_s = X.copy()
    X_s[num_feats] = scaler.transform(X[num_feats])
    X_s[num_feats] = imputer.transform(X_s[num_feats])

    # Alinear columnas al orden exacto de entrenamiento
    try:
        xgb_cols = xgb_model.get_booster().feature_names
        X_s = X_s[xgb_cols]
    except Exception:
        pass
    try:
        lgb_cols = lgb_model.booster_.feature_name()
        X_s = X_s[lgb_cols]
    except Exception:
        pass

    p_lgb = lgb_model.predict_proba(X_s)[:, 1]
    p_xgb = xgb_model.predict_proba(X_s)[:, 1]
    y_prob = best_w * p_lgb + (1 - best_w) * p_xgb
    y_pred = (y_prob >= best_t).astype(int)

    df['probabilidad_fraude'] = np.round(y_prob, 4)
    df['prediccion_fraude'] = y_pred

    if not args.no_impacto:
        df['impacto_fraude'] = [
            calcular_impacto(pred, imp)
            for pred, imp in zip(y_pred, df['importe_transaccion'])
        ]

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    detectados = y_pred.sum()
    print()
    print(f'Predicciones: {detectados} fraudes detectados ({detectados/n_total*100:.2f}%)')
    if tiene_label:
        reales = df['IS_FRAUD'].sum()
        tp = ((y_pred == 1) & (df['IS_FRAUD'] == 1)).sum()
        print(f'  TP={tp}  FN={reales - tp}  FP={detectados - tp}  TN={n_total - reales - detectados + tp}')
    print(f'Guardado: {output_path.resolve()}')


if __name__ == '__main__':
    main()
