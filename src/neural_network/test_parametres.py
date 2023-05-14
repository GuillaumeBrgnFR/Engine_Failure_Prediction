from DNN import DNN
import preprocessingData as preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

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


# function who train n model and keep the best
def train_model(X_train, y_train, X_val, y_val, nb_hidden_layer, nb_neurals, dropout, patience_stopping, epochs):
    best_model = None
    best_rmse = None
    best_r2 = None
    best_score = float('inf')
    moyenne_rmse = 0
    moyenne_r2 = 0
    moyenne_temps = 0
    nb_models = 30
    for i in range(nb_models):
        print(i)
        tps1 = time.time()
        regressor = DNN(X_train, y_train, X_val, y_val, nb_hidden_layer=nb_hidden_layer, nb_neurals=nb_neurals, dropout=dropout, patience_stopping=patience_stopping, epochs=epochs)
        tps2 = time.time()
        y_pred = regressor.predict(X_val)
        rmse = metrics.mean_squared_error(y_val, y_pred, squared=False)
        r2 = metrics.r2_score(y_val, y_pred)
        print("rmse : ", rmse)
        print("r2 : ", r2)
        moyenne_rmse += rmse
        moyenne_r2 += r2
        moyenne_temps += tps2 - tps1
        score = rmse * (1 - r2)
        if score < best_score:
            best_rmse = rmse
            best_model = regressor
            best_r2 = r2
            best_score = score

    return best_model, best_rmse, best_r2, best_score, moyenne_rmse/nb_models, moyenne_r2/nb_models, moyenne_temps/nb_models



# ready = 0 : on enlève la ligne (70k utilisable)
data = pd.read_csv("src/data/DataPHM.csv", sep=";", decimal=",")
data.dropna(inplace=True)
print(data)
data = data.drop(data[data["Ready"]==0].index)
data = data.drop(data[data["Deg"]==0].index)
data = data.dropna()
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

""" best_model, best_rmse, best_r2, best_score, moyenne_rmse, moyenne_r2, moyenne_temps = train_model(X_train, y_train, X_val, y_val, nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100)

print("best_rmse : ", best_rmse)
print("best_r2 : ", best_r2)
print("best_score : ", best_score)
print("moyenne_rmse : ", moyenne_rmse)
print("moyenne_r2 : ", moyenne_r2)
print("moyenne_temps : ", moyenne_temps)


# prediction
y_pred = best_model.predict(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=output_params)
y_pred_df.index = y_test.index

# Affichage
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison des valeurs réelles et prédites')

plot_comparison(y_test['GO'], y_pred_df['GO'], 'GO', ax, (0, 0))
plot_comparison(y_test['P1'], y_pred_df['P1'], 'P1', ax, (0, 1))
plot_comparison(y_test['PW'], y_pred_df['PW'], 'PW', ax, (1, 0))
plot_comparison(y_test['T3P'], y_pred_df['T3P'], 'T3P', ax, (1, 1))
plt.savefig(f"results/prediction_{len(input_params)}_5_500_3.png")
plt.show() """

""" best_model_GO, best_rmse_GO, best_r2_GO, best_score_GO, moyenne_rmse_GO, moyenne_r2_GO, moyenne_temps_GO = train_model(X_train, y_train["GO"], X_val, y_val["GO"], nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100)
best_model_PW, best_rmse_PW, best_r2_PW, best_score_PW, moyenne_rmse_PW, moyenne_r2_PW, moyenne_temps_PW = train_model(X_train, y_train["PW"], X_val, y_val["PW"], nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100)
best_model_T3P, best_rmse_T3P, best_r2_T3P, best_score_T3P, moyenne_rmse_T3P, moyenne_r2_T3P, moyenne_temps_T3P = train_model(X_train, y_train["T3P"], X_val, y_val["T3P"], nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100) """
best_model_P1, best_rmse_P1, best_r2_P1, best_score_P1, moyenne_rmse_P1, moyenne_r2_P1, moyenne_temps_P1 = train_model(X_train, y_train["P1"], X_val, y_val["P1"], nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=20, epochs=100)

# prediction
""" y_pred_GO = best_model_GO.predict(X_test)
y_pred_PW = best_model_PW.predict(X_test)
y_pred_T3P = best_model_T3P.predict(X_test) """
y_pred_P1 = best_model_P1.predict(X_test)

""" y_pred_df_GO = pd.Series(y_pred_GO.reshape(-1))
y_pred_df_PW = pd.Series(y_pred_PW.reshape(-1))
y_pred_df_T3P = pd.Series(y_pred_T3P.reshape(-1)) """
y_pred_df_P1 = pd.Series(y_pred_P1.reshape(-1))

y_pred_df = pd.DataFrame(columns=output_params)
""" y_pred_df["GO"] = y_pred_df_GO
y_pred_df["PW"] = y_pred_df_PW
y_pred_df["T3P"] = y_pred_df_T3P """
y_pred_df["P1"] = y_pred_df_P1
y_pred_df.index = y_test.index


print(y_pred_df)
print(y_test)


# Affichage
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison des valeurs réelles et prédites')

""" plot_comparison(y_test['GO'], y_pred_df['GO'], 'GO', ax, (0, 0))
plot_comparison(y_test['PW'], y_pred_df['PW'], 'PW', ax, (1, 0))
plot_comparison(y_test['T3P'], y_pred_df['T3P'], 'T3P', ax, (1, 1)) """
plot_comparison(y_test['P1'], y_pred_df['P1'], 'P1', ax, (0, 1))


plt.show()




""" HIDDEN_LAYERS = [2, 4, 5, 7, 10]
NEURONS = [100, 200, 500, 800, 1000]

RMSE = []
R2 = []
SCORE = []
with open("results/results.txt", "a") as file:
    for hl in HIDDEN_LAYERS:
        r2 = []
        rmse = []
        score = []
        for n in NEURONS:
            print("hidden_layer : ", hl, "neurals : ", n, "\n")
            best_model, best_rmse, best_r2, best_score, moyenne_rmse, moyenne_r2, moyenne_temps = train_model(X_train, y_train, X_val, y_val, nb_hidden_layer=hl, nb_neurals=n, dropout=0.2, patience_stopping=20, epochs=100)
            text = f"nb_hidden_layer : {hl}, nb_neurals : {n} :\nrmse : {best_rmse}\nr2 : {best_r2}\nscore : {best_score}\nmoyenne_rmse : {moyenne_rmse}\nmoyenne_r2 : {moyenne_r2}\nmoyenne_temps : {moyenne_temps}\n\n"
            file.write(text)
            best_model.save(f"results/best_models/model_{hl}_{n}.h5")
            rmse.append(best_rmse)
            r2.append(best_r2)
            score.append(best_score)
        RMSE.append(rmse)
        R2.append(r2)
        
        SCORE.append(score)

print("RMSE : ", RMSE)
print("R2 : ", R2)
print("SCORE : ", SCORE)
print("NEURONS : ", NEURONS)
print("HIDDEN_LAYERS : ", HIDDEN_LAYERS)

# plot : 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Tracé du premier graphique
for i in range(len(HIDDEN_LAYERS)):
    ax1.plot(NEURONS, RMSE[i])
    ax2.plot(NEURONS, R2[i])
    ax3.plot(NEURONS, SCORE[i])
ax1.set_title("Moyenne de la MSE en fonction du nombre de neurones.")
ax1.set_xlabel("Nombre de neurones")
ax1.set_ylabel("MSE")
ax1.legend(HIDDEN_LAYERS, loc='upper right', title="Couche cachées")
ax2.set_title("Moyenne du R2 en fonction du nombre de neurones.")
ax2.set_xlabel("Nombre de neurones")
ax2.set_ylabel("R2")
ax2.legend(HIDDEN_LAYERS, loc='upper right', title="Couche cachées")
ax3.set_title("Moyenne du score (RMSE * (1-R2)) en fonction du nombre de neurones.")
ax3.set_xlabel("Nombre de neurones")
ax3.set_ylabel("Score")
ax3.legend(HIDDEN_LAYERS, loc='upper right', title="Couche cachées")

fig.set_size_inches(25, 12)

plt.savefig("results/test_params.png")
plt.show() """
