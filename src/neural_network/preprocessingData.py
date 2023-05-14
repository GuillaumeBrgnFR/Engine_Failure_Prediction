import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PreprocessingDNN:

    # train set
    X_train = None
    y_train = None

    # test set
    X_test = None
    y_test = None

    # validation data
    X_val = None
    y_val = None

    # standardize data
    scaler = None



    def __init__(self, df : pd.DataFrame):

        # input and output 
        input_params = ["Deg", "Mo", "CO", "CR", "T1"]
        output_params = ["GO", "P1", "PW", "T3P"]

        X = df[input_params]
        y = df[output_params]

        # standardize data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # train set
        split = int(0.8*len(df))
        self.X_train = X[:split, :]
        self.X_test = X[split:, :]
        self.y_train = y.iloc[:split,:]
        self.y_test = y.iloc[split:,:]


        ##self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        split_val = int(0.8*self.X_train.shape[0])
        self.X_val = self.X_train[split_val:, :]
        self.X_train = self.X_train[:split_val, :]
        self.y_val = self.y_train.iloc[split_val:,:]
        self.y_train = self.y_train.iloc[:split_val,:]
        
        #self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, shuffle=True)

        print("DataFrame.shape : ", df.shape)
        print("X_train shape: ", self.X_train.shape)
        print("y_train shape: ", self.y_train.shape, " index_start : ", self.y_train.index[0], " index_end : ", self.y_train.index[-1])
        print("X_test shape: ", self.X_test.shape)
        print("y_test shape: ", self.y_test.shape, " index_start : ", self.y_test.index[0], " index_end : ", self.y_test.index[-1])
        print("X_val shape: ", self.X_val.shape)
        print("y_val shape: ", self.y_val.shape, " index_start : ", self.y_val.index[0], " index_end : ", self.y_val.index[-1])

        try :
            self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1]))
            self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1]))
            self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], self.X_val.shape[1]))
        except :
            self.X_train = self.X_train.reshape(-1, 1)
            self.X_test = self.X_test.reshape(-1, 1)
            self.X_val = self.X_val.reshape(-1, 1)
        

    def getXtrain(self):
        return self.X_train
    
    def getYtrain(self):
        return self.y_train
    
    def getXtest(self):
        return self.X_test
    
    def getYtest(self):
        return self.y_test
    
    def getXtrainScaled(self):
        return self.X_train_scaled
    
    def getXtestScaled(self):
        return self.X_test_scaled

    def getScaler(self):
        return self.scaler

    def getXval(self):
        return self.X_val
    
    def getYval(self):
        return self.y_val

    def getXvalScaled(self):
        return self.X_val_scaled
