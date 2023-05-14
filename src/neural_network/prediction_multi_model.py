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
data = data.drop(data[data["Ready"]==0].index)
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

HL = 4
NN = 600
PS = 15
EPOCHS = 200

# DNN
regressor_GO = DNN(X_train, y_train["GO"], X_val, y_val["GO"], nb_hidden_layer=HL, nb_neurals=NN, dropout=0.2, patience_stopping=PS, epochs=EPOCHS)
regressor_PW = DNN(X_train, y_train["PW"], X_val, y_val["PW"], nb_hidden_layer=HL, nb_neurals=NN, dropout=0.2, patience_stopping=PS, epochs=EPOCHS)
regressor_T3P = DNN(X_train, y_train["T3P"], X_val, y_val["T3P"], nb_hidden_layer=HL, nb_neurals=NN, dropout=0.2, patience_stopping=PS, epochs=EPOCHS)
regressor_P1 = DNN(X_train, y_train["P1"], X_val, y_val["P1"], nb_hidden_layer=HL, nb_neurals=NN, dropout=0.2, patience_stopping=PS, epochs=EPOCHS)

# prediction
y_pred_GO = regressor_GO.predict(X_test)
y_pred_PW = regressor_PW.predict(X_test)
y_pred_T3P = regressor_T3P.predict(X_test)
y_pred_P1 = regressor_P1.predict(X_test)

y_pred_df_GO = pd.Series(y_pred_GO.reshape(-1))
y_pred_df_PW = pd.Series(y_pred_PW.reshape(-1))
y_pred_df_T3P = pd.Series(y_pred_T3P.reshape(-1))
y_pred_df_P1 = pd.Series(y_pred_P1.reshape(-1))

y_pred_df = pd.DataFrame(columns=output_params)
y_pred_df["GO"] = y_pred_df_GO
y_pred_df["PW"] = y_pred_df_PW
y_pred_df["T3P"] = y_pred_df_T3P
y_pred_df["P1"] = y_pred_df_P1





# evaluation
""" print("\n")
print("RMSE_GO : ", metrics.mean_squared_error(y_test["GO"], y_pred_df["GO"], squared=False))
print("R2_GO : ", metrics.r2_score(y_test["GO"], y_pred_df["GO"]))
print("\n")
print("RMSE_PW : ", metrics.mean_squared_error(y_test["PW"], y_pred_df["PW"], squared=False))
print("R2_PW : ", metrics.r2_score(y_test["PW"], y_pred_df["PW"]))
print("\n")
print("RMSE_T3P : ", metrics.mean_squared_error(y_test["T3P"], y_pred_df["T3P"], squared=False))
print("R2_T3P : ", metrics.r2_score(y_test["T3P"], y_pred_df["T3P"]))
print("\n") """
print("RMSE_P1 : ", metrics.mean_squared_error(y_test["P1"], y_pred_df["P1"], squared=False))
print("R2_P1 : ", metrics.r2_score(y_test["P1"], y_pred_df["P1"]))
print("\n")


# Affichage
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison des valeurs réelles et prédites')

plot_comparison(y_test['GO'], y_pred_df['GO'], 'GO', ax, (0, 0))
plot_comparison(y_test['PW'], y_pred_df['PW'], 'PW', ax, (1, 0))
plot_comparison(y_test['T3P'], y_pred_df['T3P'], 'T3P', ax, (1, 1))
plot_comparison(y_test['P1'], y_pred_df['P1'], 'P1', ax, (0, 1))


plt.show()
