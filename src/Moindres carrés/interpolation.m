%% Chargement et visualisation des donnees
clear all
M0= xlsread('../data/DataPHM.xlsx','A1:J90850'); % Chargement de données à partir du fichier exel
M=[];
n=size(M0,1); % Nombre de ligne de M0
%% Nettoyage de données
j=1;
for i=1:n
    if (M0(i,10)==1) % Ready =1 ou etat de marche
        M(j,:)=M0(i,:);
        j=j+1;
    end
end
%% Tracer

hold on; plot(100*M(:,1));
hold on; plot(M(:,1));
hold on; plot(100*M(:,1));
hold on; plot(100*M(:,1));
 %% T3P comme indicateur de santé
col_T3P_CO=corrcoef(M(:,8),M(:,4)) % Correlation entre T3P et CO
col_T3P_CR=corrcoef(M(:,8),M(:,5)) % Correlation entre T3P et CO
col_T3P_T1=corrcoef(M(:,8),M(:,9)) % Correlation entre T3P et CO
col_T3P_Mo=corrcoef(M(:,8),M(:,2)) % Correlation entre T3P et CO
%% Recherche Temps de passage 
Tpass=1;
while(M(Tpass,1)==0)
    Tpass=Tpass+1;
end
Tpass

%% Régression linéaire

% Calculer les coefficients de la régression linéaire pour chaque capteur et chaque mode opératoire
p_GO = polyfit(M(:, 4), M(:, 3), 1);
p_P1 = polyfit(M(:, 4), M(:, 6), 1);
p_PW = polyfit(M(:, 4), M(:, 7), 1);
p_T3P = polyfit(M(:, 4), M(:, 8), 1);

% Calculer les valeurs estimées pour chaque capteur à l'aide de la régression linéaire
GO_esti = polyval(p_GO, M(:, 4));
P1_esti = polyval(p_P1, M(:, 4));
PW_esti = polyval(p_PW, M(:, 4));
T3P_esti = polyval(p_T3P, M(:, 4));

%% Calculer les valeurs de RMSE et R^2 pour chaque capteur
[r2_GO, rmse_GO] = rsquare(M(:, 3)', GO_esti');
[r2_P1, rmse_P1] = rsquare(M(:, 6)', P1_esti');
[r2_PW, rmse_PW] = rsquare(M(:, 7)', PW_esti');
[r2_T3P, rmse_T3P] = rsquare(M(:, 8)', T3P_esti');





%% Calculer les valeurs de RMSE et R^2 pour chaque capteur
%[r2_GO, rmse_GO] = rsquare(M(Tpass+1:end, 3)', GO_esti(Tpass+1:end)');
%[r2_P1, rmse_P1] = rsquare(M(Tpass+1:end, 6)', P1_esti(Tpass+1:end)');
%[r2_PW, rmse_PW] = rsquare(M(Tpass+1:end, 7)', PW_esti(Tpass+1:end)');
%[r2_T3P, rmse_T3P] = rsquare(M(Tpass+1:end, 8)', T3P_esti(Tpass+1:end)');
% Afficher les graphiques comparatifs pour chaque capteur
figure;
subplot(2, 2, 1);
plot(M(Tpass+1:end, 3), 'b'); hold on;
plot(GO_esti(Tpass+1:end), 'r');
xlabel('Temps');
ylabel('GO');
title('Comparaison des valeurs réelles et interpolées de GO');
legend('Valeurs réelles', 'Valeurs interpolées');
text(0.5, 0.5, sprintf('RMSE = %.2f\nR^2 = %.2f', rmse_GO, r2_GO), 'Units', 'normalized');

subplot(2, 2, 2);
plot(M(Tpass+1:end, 6), 'b'); hold on;
plot(P1_esti(Tpass+1:end), 'r');
xlabel('Temps');
ylabel('P1');
title('Comparaison des valeurs réelles et interpolées de P1');
legend('Valeurs réelles', 'Valeurs interpolées');
text(0.5, 0.5, sprintf('RMSE = %.2f\nR^2 = %.2f', rmse_P1, r2_P1), 'Units', 'normalized');

subplot(2, 2, 3);
plot(M(Tpass+1:end, 7), 'b'); hold on;
plot(PW_esti(Tpass+1:end), 'r');
xlabel('Temps');
ylabel('PW');
title('Comparaison des valeurs réelles et interpolées de PW');
legend('Valeurs réelles', 'Valeurs interpolées');
text(0.5, 0.5, sprintf('RMSE = %.2f\nR^2 = %.2f', rmse_PW, r2_PW), 'Units', 'normalized');

subplot(2, 2, 4);
plot(M(Tpass+1:end, 8), 'b'); hold on;
plot(T3P_esti(Tpass+1:end), 'r');
xlabel('Temps');
ylabel('T3P');
title('Comparaison des valeurs réelles et interpolées de T3P');
legend('Valeurs réelles', 'Valeurs interpolées');
text(0.5, 0.5, sprintf('RMSE = %.2f\nR^2 = %.2f', rmse_T3P, r2_T3P), 'Units', 'normalized');
