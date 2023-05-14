%% Chargement et visualisation des donnees
clear all
M0= xlsread('DataPHM.xlsx','A1:J90850'); % Chargement de données à partir du fichier exel
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
% Sorties possibles: M(:,3)G0,M(:,6)P1,M(:,7)PW,M(:,8)T3P
%plot(M(:,1)); xlabel('Temps'); ylabel('Mode ');title('Mode de dégradation');
%plot(M(:,2));xlabel('Temps'); ylabel('Mode opératoire'); title('Mode opératoire');
plot(M(:,3));xlabel('Temps'); ylabel('GO'); title('Sensor GO');
hold on; plot(100*M(:,1));
%figure;stairs(M(:,4));xlabel('Temps'); ylabel('CO'); title('Sensor CO');
%figure; plot(M(:,5));xlabel('Temps'); ylabel('CR'); title('Sensor CR');
figure; plot(M(:,6));xlabel('Temps'); ylabel('P1');title('Sensor P1');
hold on; plot(M(:,1));
figure; plot(M(:,7));xlabel('Temps'); ylabel('PW');title('Sensor PW');
hold on; plot(100*M(:,1));
figure; plot(M(:,8));xlabel('Temps'); ylabel('T3P'); title('Sensor T3P');
hold on; plot(100*M(:,1));
%figure; plot(M(:,9));xlabel('Temps'); ylabel('T1'); title('Sensor T1');
%figure; plot(M(:,10));xlabel('Temps'); ylabel('Ready');title('Ready');
 %% T3P comme indicateur de santé
  % Recherche des entrées plus significatives:
  % M0 (M:,2),CO M(:,4),CR M(:,5),T1 M(:,9)
col_T3P_CO=corrcoef(M(:,8),M(:,4)) % Correlation entre T3P et CO
col_T3P_CR=corrcoef(M(:,8),M(:,5)) % Correlation entre T3P et CO
col_T3P_T1=corrcoef(M(:,8),M(:,9)) % Correlation entre T3P et CO
col_T3P_Mo=corrcoef(M(:,8),M(:,2)) % Correlation entre T3P et CO
% Selectioner MO et CO comme entrees:  T3P=f(MO,CO) ?
%% Recherche Temps de passage 
Tpass=1;
while(M(Tpass,1)==0)
    Tpass=Tpass+1;
end
Tpass
%% Regression
X1=[];X2=[];
Y1=[];Y2=[]; 
X=[];Y=[];
T_esti=round(Tpass*0.7);
for i=1:T_esti
    X(i)=M(i,4);Y(i)=M(i,8);
    if (M(i,2)==1)% Mode opératoire 1
        X1=[X1 M(i,4)];
        Y1=[Y1 M(i,8)];
    else % Mode operatoire 2
        X2=[X2 M(i,4)];
        Y2=[Y2 M(i,8)];
    end
end
f1=[];f1=polyfit(X1,Y1,1)% regestion non-linénaire de dégré 2: Y1=f1(1)*X1^2+f1(2)*X1+f1(3);
f2=[];f2=polyfit(X2,Y2,1)% regestion non-linénaire de dégré 2: Y2=f2(1)*X2^2+f2(2)*X2+f2(3);
f=[];f=polyfit(X,Y,1)% regestion non-linénaire de dégré 2: Y=f(1)*X^2+f(2)*X2+f(3);
%% Test
T3P_esti=[];
T3P_esti_sansmode=[];
for i=1:Tpass
    T3P_esti_sansmode(i)=polyval(f,M(i,4));
    if (M(i,2)==1) % Mode opératoire 1
        T3P_esti(i)=polyval(f1,M(i,4));
    else % Mode opératoire 2
        T3P_esti(i)=polyval(f2,M(i,4));
    end
end

[r2 rmse] = rsquare(M(T_esti+1:Tpass,8)',T3P_esti(T_esti+1:Tpass)) 

