from DNN import DNN
import preprocessingData as preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics


def plot_comparison(y_true, y_pred, y_label, ax, index):
    ax[index].plot(y_true, label='Valeurs réelles')
    ax[index].plot(y_pred, label='Valeurs prédites', linestyle='--')
    ax[index].set_xlabel('Index')
    ax[index].set_ylabel(y_label)
    r2 = metrics.r2_score(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    ax[index].set_title(f'{y_label} - R²: {r2:.2f}, RMSE: {rmse:.2f}')
    ax[index].legend()


# ready = 0 : on enlève la ligne (70k utilisable)

data = pd.read_csv("src/data/DataPHM.csv", sep=";", decimal=",")
data.dropna(inplace=True)
print(data)
data = data.drop(data[data["Ready"]==0].index)
data = data.drop(data[data["Deg"]==0].index)
data = data.dropna()
#data.index = range(len(data))
print("data.shape : ", data.shape)
print("data.head : \n", data.head())

# matrice de corrélation
""" corr = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt=".2f") """
#plt.show()corr = data.corr()


dict_params = {
    "Deg" : "Degrade 1 ou 0",
    "Mo" : "Mode de fonctionnement (1 ou 2)",
    "GO" : "Consommation carburant (l/h)",
    "CO" : "Couple moteur (%)",
    "CR" : "Position cran moteur (%)",
    "P1" : "Pression de la suralimentation (bar)",
    "PW" : "Puissance moteur (%)",
    "T3P" : "Temperature sortie compresseur (°C)",
    "T1" : "Temperature d'admission (°C)",
    "Ready" : "1 si le moteur fonctionne"
}


input_params = ["Deg", "Mo", "CO", "CR", "T1"]
output_params = ["GO", "P1", "PW", "T3P"]


# preprocessing
preprocessing = preprocessing.PreprocessingDNN(data)
X_train = preprocessing.getXtest()
y_train = preprocessing.getYtest()
X_test = preprocessing.getXval()
y_test = preprocessing.getYval()
X_val = preprocessing.getXval()
y_val = preprocessing.getYval()

#y_test = y_test.reset_index(drop=True)

print(y_test)


# DNN
regressor = DNN(X_train, y_train, X_val, y_val, nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100)

# prediction
y_pred = regressor.predict(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=output_params)
y_pred_df.index = y_test.index



# evaluation
print("\n")
print("RMSE_tot : ", metrics.mean_squared_error(y_test, y_pred, squared=False))
print("R2_tot : ", metrics.r2_score(y_test, y_pred))
print("\n")
print("RMSE_GO : ", metrics.mean_squared_error(y_test["GO"], y_pred_df["GO"], squared=False))
print("R2_GO : ", metrics.r2_score(y_test["GO"], y_pred_df["GO"]))
print("\n")
print("RMSE_P1 : ", metrics.mean_squared_error(y_test["P1"], y_pred_df["P1"], squared=False))
print("R2_P1 : ", metrics.r2_score(y_test["P1"], y_pred_df["P1"]))
print("\n")
print("RMSE_PW : ", metrics.mean_squared_error(y_test["PW"], y_pred_df["PW"], squared=False))
print("R2_PW : ", metrics.r2_score(y_test["PW"], y_pred_df["PW"]))
print("\n")
print("RMSE_T3P : ", metrics.mean_squared_error(y_test["T3P"], y_pred_df["T3P"], squared=False))
print("R2_T3P : ", metrics.r2_score(y_test["T3P"], y_pred_df["T3P"]))
print("\n")


# Affichage
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison des valeurs réelles et prédites')

plot_comparison(y_test['GO'], y_pred_df['GO'], 'GO', ax, (0, 0))
plot_comparison(y_test['P1'], y_pred_df['P1'], 'P1', ax, (0, 1))
plot_comparison(y_test['PW'], y_pred_df['PW'], 'PW', ax, (1, 0))
plot_comparison(y_test['T3P'], y_pred_df['T3P'], 'T3P', ax, (1, 1))
plt.savefig(f"results/prediction_{len(input_params)}inputs_degrade_1__4.png")
plt.show()
