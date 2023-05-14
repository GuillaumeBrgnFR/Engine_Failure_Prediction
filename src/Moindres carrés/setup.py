import pandas as pd

input_file = 'DataPHM.xlsx'
output_file = 'DataPHM1.xlsx'

# Lire le fichier Excel et charger dans un DataFrame
df = pd.read_excel(input_file)

# Supprimer les lignes où la colonne 'Ready' a une valeur de 0
df = df[df['Ready'] != 0]

# Sauvegarder le DataFrame modifié dans un nouveau fichier Excel
df.to_excel(output_file, index=False)