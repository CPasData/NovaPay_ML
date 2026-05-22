import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

df_real = pd.read_csv('dataset_fraude.csv', parse_dates=['fecha_hora', 'fecha_creacion_tarjeta'])

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_real)

metadata.update_column('id_cliente', sdtype='id')
metadata.update_column('id_cuenta', sdtype='id')
metadata.update_column('id_tarjeta', sdtype='id')
metadata.update_column('id_transaccion', sdtype='id')
metadata.update_column('fecha_hora', sdtype='datetime')
metadata.update_column('fecha_creacion_tarjeta', sdtype='datetime')
metadata.update_column('identificador_dispositivo_fingerprint', sdtype='categorical')
metadata.update_column('direccion_ip_origen', sdtype='categorical')
metadata.update_column('geolocalizacion', sdtype='categorical')
metadata.update_column('customer_country', sdtype='categorical')
metadata.update_column('customer_region', sdtype='categorical')
metadata.update_column('tipo_cliente', sdtype='categorical')
metadata.update_column('estado_cuenta', sdtype='categorical')
metadata.update_column('estado_tarjeta', sdtype='categorical')
metadata.update_column('metodo_autenticacion', sdtype='categorical')
metadata.update_column('IS_FRAUD', sdtype='categorical')

print("Metadatos:")
print(metadata)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df_real)

synthetic_data = synthesizer.sample(num_rows=10000)

synthetic_data.to_csv('dataset_fraude_sdv.csv', index=False, encoding='utf-8')
print(f"\nDatos sintéticos generados: {len(synthetic_data)} filas")
print("Guardado en dataset_fraude_sdv.csv")

print("\nPrimeras 5 filas:")
print(synthetic_data.head())
