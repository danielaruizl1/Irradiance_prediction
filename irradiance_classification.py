import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pedir al usuario el nombre del archivo CSV
sensor_xlsx = "sensors20230822.xlsx"

# Cargar el archivo CSV en un DataFrame
try:
    df = pd.read_excel(sensor_xlsx)
except FileNotFoundError:
    print(f"El archivo {sensor_xlsx} no fue encontrado.")
    exit()

df['Hour_str'] = df['Hour corrected'].apply(lambda x: x.strftime('%H:%M:%S'))
df_short = df
#df_short = df.iloc[603::4]
i_labels = [df_short.index[0], df_short.index[-1]]
show_labels = [df_short['Hour_str'][i] for i in i_labels]

# Crear la gráfica
plt.figure(figsize=(12, 6))
#plt.plot(df_short["Hour_str"], df_short["ghi1"], marker='o', linestyle='-')
sns.lineplot(x='Hour_str', y='ghi1', data=df_short, marker='o', color='yellow', markersize=8, markeredgecolor='yellow', linestyle='-')
plt.xticks([0,len(df_short)], show_labels)
plt.title("Irradiación a lo largo de un día (20230822)")
plt.xlabel("Hora")
plt.ylabel("GHI (W/m2)")
plt.grid(True)

# Mostrar la gráfica
plt.savefig('irradiance0822.png')
plt.show()
