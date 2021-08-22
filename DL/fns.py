import numpy as np
import pandas as pd
#import hvplot.pandas
from sklearn.preprocessing import MinMaxScaler

# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

# Define a Splitter
def train_test_splitter(df:pd.DataFrame=None, percent_data=0.7):
    """Manually Splits data into training and testing sets

    Args:
        percent_data (float, optional): The percentage of data to keep. Defaults to 0.7.
    """
    mty_df = pd.DataFrame()
    Nothing = False, mty_df, mty_df, mty_df, mty_df

    if df is None:
        return Nothing
    
    if percent_data>100:
        return  Nothing
    elif percent_data>1.0 and percent_data<=100.0:
        percent_data = percent_data/100.00

    split = int(percent_data * len(df))
    print(f"Of the {len(df)} records {split} are used for training leaving {len(df) - split} for testing")

    X_train     = df[:split]
    X_test      = df[split:]
    y_train     = df[:split]
    y_test      = df[split:]

    return True, X_train, X_test, y_train, y_test

class MinMaxOperator:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        self.x_train_scaler  = MinMaxScaler()
        self.x_test_scaler   = MinMaxScaler()
        self.y_train_scaler  = MinMaxScaler()
        self.y_test_scaler   = MinMaxScaler()

    def pprint(self):
        for x in [self.X_train, self.X_test, self.y_train, self.y_test]:
            print(f"{x[:1]} has shape {x.shape}")

    def MinMaxScaler_Operation(self):    #X_train:NDFrame,X_test:NDFrame,y_train:NDFrame,y_test:NDFrame):
        # Fit the scaler for the Training Data
        self.x_train_scaler.fit(self.X_train)
        self.y_train_scaler.fit(self.y_train)

        # Scale the training data
        X_train_sc = self.x_train_scaler.transform(self.X_train)
        y_train_sc = self.y_train_scaler.transform(self.y_train)

        # Fit the scaler for the Testing Data
        self.x_test_scaler.fit(self.X_test)
        self.y_test_scaler.fit(self.y_test)

        # Scale the y_test data
        X_test_sc = self.x_test_scaler.transform(self.X_test)
        y_test_sc = self.y_test_scaler.transform(self.y_test)

        return X_train_sc,X_test_sc,y_train_sc,y_test_sc    #, y_test_scaler

def LSTM_Reshaper(df:pd.DataFrame):
    if len(df.shape)==3:
        return df
    rows, cols = df.shape
    return df.reshape(rows, cols, 1)

if __name__ == "__main__":

    # Load the fear and greed sentiment data for Bitcoin
    df = pd.read_csv(
        """D:/Data/Family/Ethol Palmer/UoT SCS/FinTech/Repos/Deep-Learning-homework/btc_sentiment.csv"""
        , index_col="date"
        , infer_datetime_format=True
        , parse_dates=True
    )
    df = df.drop(columns="fng_classification")
    print(df.head())

    # Load the historical closing prices for Bitcoin
    df2 = pd.read_csv(
        """D:/Data/Family/Ethol Palmer/UoT SCS/FinTech/Repos/Deep-Learning-homework/btc_historic.csv"""
        , index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
    df2 = df2.sort_index()
    df2.tail()
    print(df2.head())

    # Join the data into a single DataFrame
    df = df.join(df2, how="inner")
    print(df.tail())

    feature_column  = 1
    target_column   = 1
    X, y = window_data(df=df, window=10, feature_col_number=feature_column, target_col_number=target_column)
    print(X.shape, X[:4])
    print(y.shape, y[:4])

    success, X_train, X_test, y_train, y_test = train_test_splitter(df=X)

    trainer_tester = MinMaxOperator(X_train, X_test, y_train, y_test)
    trainer_tester.pprint()
    X_train_sc,X_test_sc,y_train_sc,y_test_sc = trainer_tester.MinMaxScaler_Operation()
    print(X_train_sc[:4])

    X_train_sc = LSTM_Reshaper(X_train_sc)
    X_test_sc  = LSTM_Reshaper(X_test_sc)

