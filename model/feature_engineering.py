import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


ID_COLS = [
    'id_cliente', 'id_cuenta', 'cuenta_origen', 'id_tarjeta',
    'id_transaccion', 'identificador_dispositivo_fingerprint',
    'direccion_ip_origen', 'geolocalizacion', 'cuenta_destino',
    'fecha_creacion_tarjeta', 'fecha_hora',
]

DROP_COLS = ID_COLS + ['numero_fraudes_ultimo_ano']

CAT_COLS = [
    'tipo_cliente', 'customer_country', 'customer_region',
    'estado_cuenta', 'estado_tarjeta', 'tipo_transaccion',
    'metodo_autenticacion', 'operacion_pais', 'operacion_region',
]

NUM_COLS = [
    'edad_cliente', 'tenure', 'importe_medio_mensual',
    'desviacion_estandar_mensual', 'media_transacciones_al_dia',
    'saldo_actual', 'saldo_medio_30_dias',
    'volumen_entrante_30_dias', 'volumen_saliente_30_dias',
    'numero_transferencias_recibidas_7_dias',
    'numero_transferencias_enviadas_7_dias',
    'antiguedad_tarjeta_dias', 'limite_importe_transacciones',
    'veces_superar_limite_7_dias', 'is_night', 'is_weekend',
    'tiempo_desde_ultima_transaccion',
    'numero_transacciones_ultima_hora', 'importe_transaccion',
    'numero_pin_disponibles', 'dispositivo_reconocido',
    'destino_alto_riesgo',
]

TARGET_BINARY = 'IS_FRAUD'
TARGET_IMPACTO = 'IMPACTO_FRAUDE'


def _safe_ratio(a, b, fill=0):
    denom = np.where(b == 0, np.nan, b)
    result = np.divide(a, denom, out=np.full_like(a, fill, dtype=float), where=denom != 0)
    return np.nan_to_num(result, nan=fill)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, encode_target=None, n_folds=5, random_state=42):
        self.encode_target = encode_target
        self.n_folds = n_folds
        self.random_state = random_state
        self._freq_encodings = {}
        self._target_encodings = {}
        self._global_means = {}
        self._fitted = False

    def fit(self, X, y=None):
        X = X.copy()
        self._freq_encodings = {}
        self._target_encodings = {}
        self._fitted = True

        for col in CAT_COLS:
            if col in X.columns:
                self._freq_encodings[col] = X[col].value_counts().to_dict()

        if self.encode_target is not None:
            if y is None and self.encode_target in X.columns:
                y = X[self.encode_target]
            if y is not None:
                self._global_means[self.encode_target] = y.mean()
                for col in CAT_COLS:
                    if col in X.columns:
                        self._target_encodings[col] = (
                            X.assign(_tmp_target=y).groupby(col)['_tmp_target'].mean().to_dict()
                        )

        return self

    def transform(self, X):
        X = X.copy()

        X['hour'] = pd.to_datetime(X['fecha_hora']).dt.hour if 'fecha_hora' in X.columns else 0
        X['day_of_week'] = pd.to_datetime(X['fecha_hora']).dt.dayofweek if 'fecha_hora' in X.columns else 0
        X['is_weekday'] = (~X['is_weekend'].astype(bool)).astype(int) if 'is_weekend' in X.columns else 1
        X.drop(columns=['fecha_hora'], inplace=True, errors='ignore')

        for col in DROP_COLS:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        X['txn_vs_limit_pct'] = X['importe_transaccion'] / X['limite_importe_transacciones'].replace(0, 1)
        X['txn_vs_balance_pct'] = _safe_ratio(X['importe_transaccion'], X['saldo_actual'])
        X['txn_vs_monthly_avg_pct'] = _safe_ratio(X['importe_transaccion'], X['importe_medio_mensual'])
        X['balance_vs_avg_pct'] = _safe_ratio(X['saldo_actual'], X['saldo_medio_30_dias'])
        X['outflow_inflow_ratio'] = _safe_ratio(X['volumen_saliente_30_dias'], X['volumen_entrante_30_dias'])
        X['net_flow_30d'] = X['volumen_entrante_30_dias'] - X['volumen_saliente_30_dias']
        X['limite_breach_rate'] = X['veces_superar_limite_7_dias'] / 7.0
        X['txn_intensity'] = X['numero_transacciones_ultima_hora'] / (X['tiempo_desde_ultima_transaccion'] + 1)
        X['balance_utilization'] = X['saldo_medio_30_dias'] / (X['limite_importe_transacciones'] * 10 + 1)
        X['txn_severity'] = X['importe_transaccion'] * X['numero_transacciones_ultima_hora']
        X['tenure_years'] = X['tenure'] / 365.0

        if 'customer_country' in X.columns and 'operacion_pais' in X.columns:
            X['cross_border'] = (X['customer_country'] != X['operacion_pais']).astype(int)
        if 'customer_region' in X.columns and 'operacion_region' in X.columns:
            X['cross_region'] = (X['customer_region'] != X['operacion_region']).astype(int)
        if 'customer_country' in X.columns and 'operacion_pais' in X.columns:
            X['same_country_device'] = (
                (X['customer_country'] == X['operacion_pais']) & (X['dispositivo_reconocido'] == 1)
            ).astype(int)

        X['importe_transaccion_log'] = np.log1p(X['importe_transaccion'])
        X['saldo_actual_log'] = np.log1p(X['saldo_actual'])
        X['volumen_saliente_log'] = np.log1p(X['volumen_saliente_30_dias'])
        X['volumen_entrante_log'] = np.log1p(X['volumen_entrante_30_dias'])
        X['tiempo_ultima_log'] = np.log1p(X['tiempo_desde_ultima_transaccion'])

        for col in CAT_COLS:
            if col in X.columns:
                if col in self._freq_encodings:
                    X[f'{col}_freq'] = X[col].map(self._freq_encodings[col]).fillna(0)
                if self.encode_target is not None and col in self._target_encodings:
                    mean = self._global_means.get(self.encode_target, 0)
                    X[f'{col}_target'] = X[col].map(self._target_encodings[col]).fillna(mean)
                X.drop(columns=[col], inplace=True, errors='ignore')

        for col in self._freq_encodings:
            if f'{col}_freq' not in X.columns:
                X[f'{col}_freq'] = 0
        for col in self._target_encodings:
            if f'{col}_target' not in X.columns:
                X[f'{col}_target'] = self._global_means.get(self.encode_target, 0)

        return X

    def get_feature_names_out(self, input_features=None):
        return None


def engineer_features(df, target_col=None, fit=True, transformer=None):
    if fit:
        y = df[target_col] if target_col else None
        transformer = FeatureEngineer(encode_target=target_col)
        transformer.fit(df, y)
        result = transformer.transform(df)
        return result, transformer
    else:
        if transformer is None:
            raise ValueError("Must provide a fitted transformer when fit=False")
        result = transformer.transform(df)
        return result


def prepare_train_test_split(df, target_col, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])
    return train, test
