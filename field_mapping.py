# ============================================================
# FIELD MAPPING — NovaPay ML
# Mapeo de nombres entre el CSV/FeatureEngineer
# y la BD (nombres cortos)
# ============================================================
# Uso:
#   from field_mapping import CSV_A_BD, BD_A_CSV, FE_A_BD
#
#   # Renombrar columnas del CSV para insertar en BD
#   df.rename(columns=CSV_A_BD, inplace=True)
#
#   # Renombrar columnas de BD para usar en el modelo
#   df.rename(columns=BD_A_CSV, inplace=True)
#
#   # Renombrar campos calculados por FeatureEngineer para BD
#   campos_fe = {k: v for k, v in FE_A_BD.items() if k in df.columns}
#   df.rename(columns=campos_fe, inplace=True)
# ============================================================


# ── TABLA CLIENTES ──────────────────────────────────────────
CSV_A_BD_CLIENTES = {
    "customer_country"              : "pais_residencia",
    "customer_region"               : "region_residencia",
    "tenure"                        : "permanencia",
    "importe_medio_mensual"         : "import_avg_mensual",
    "desviacion_estandar_mensual"   : "import_std_mensual",
    "media_transacciones_al_dia"    : "avg_tx_dia",
    "numero_fraudes_ultimo_ano"     : "num_fraudes_anio",
}


# ── TABLA TRANSACCIONES ─────────────────────────────────────
CSV_A_BD_TRANSACCIONES = {
    "saldo_medio_30_dias"                   : "saldo_avg_30d",
    "volumen_entrante_30_dias"              : "vol_entrante_30d",
    "volumen_saliente_30_dias"              : "vol_saliente_30d",
    "numero_transferencias_recibidas_7_dias": "num_trans_recib_7d",
    "numero_transferencias_enviadas_7_dias" : "num_trans_env_7d",
    "antiguedad_tarjeta_dias"               : "tiempo_activo_tarjeta",
    "limite_importe_transacciones"          : "limite_impot_tx",
    "veces_superar_limite_7_dias"           : "veces_superar_limite_7d",
    "fecha_hora"                            : "fecha_trans",
    "tiempo_desde_ultima_transaccion"       : "tiempo_desde_ult_trans",
    "numero_transacciones_ultima_hora"      : "num_trans_ult_hora",
    "metodo_autenticacion"                  : "metod_auten",
    "numero_pin_disponibles"                : "num_veces_pin",
    "identificador_dispositivo_fingerprint" : "ident_disp",
    "direccion_ip_origen"                   : "dir_ip_origen",
    "IS_FRAUD"                              : "is_fraud",
    "IMPACTO_FRAUDE"                        : "impacto_fraude",
}


# ── CAMPOS CALCULADOS POR FEATUREENGINEER → BD ──────────────
# No vienen del CSV, los calcula el FeatureEngineer
# Se extraen ANTES de escalar y se guardan en BD
FE_A_BD = {
    "cross_border"      : "es_transfronteriza",
    "txn_vs_limit_pct"  : "ratio_imp_limite",
    "txn_intensity"     : "intensidad_tx",
    "txn_severity"      : "severidad_tx",
    "net_flow_30d"      : "flujo_neto_30d",
}


# ── MAPEO COMPLETO CSV → BD ─────────────────────────────────
CSV_A_BD = {**CSV_A_BD_CLIENTES, **CSV_A_BD_TRANSACCIONES}


# ── MAPEO INVERSO BD → CSV ──────────────────────────────────
BD_A_CSV = {v: k for k, v in CSV_A_BD.items()}


# ── CAMPOS VACÍOS AL GENERAR EL CSV ─────────────────────────
# Los rellena el modelo automáticamente
CAMPOS_MODELO = [
    "is_fraud",
    "prob_fraud",
    "es_transfronteriza",
    "ratio_imp_limite",
    "intensidad_tx",
    "severidad_tx",
    "flujo_neto_30d",
]

# Los rellena el analista desde la app de Full Stack
CAMPOS_ANALISTA = [
    "target_final",
    "estado_revision",
    "id_usuario",
    "fecha_revision",
]

CAMPOS_VACIOS_AL_GENERAR = CAMPOS_MODELO + CAMPOS_ANALISTA
