import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data'
df_real = pd.read_csv(data_dir / 'dataset_fraude.csv', parse_dates=['fecha_hora', 'fecha_creacion_tarjeta'])

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

output_path = data_dir / 'dataset_fraude_sdv.csv'
synthetic_data.to_csv(output_path, index=False, encoding='utf-8')
print(f"\nDatos sintéticos generados: {len(synthetic_data)} filas")
print(f"Guardado en {output_path}")

print("\nPrimeras 5 filas:")
print(synthetic_data.head())
