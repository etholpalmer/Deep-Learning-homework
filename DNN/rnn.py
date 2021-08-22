from pandas.core.algorithms import mode
from pandas.core.generic import NDFrame
import numpy as np
import pandas as pd
import hvplot.pandas
from sklearn.preprocessing import MinMaxScaler

def window_data(df, window, feature_col_number, target_col_number):
    """ This function accepts the column number for the features (X) and the target (y)
        It chunks the data up with a rolling window of Xt-n to predict Xt
        It returns a numpy array of X any y
    """
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

def Pct_Converter(val, default_val=0.70):
    """Checks if a value passed is a decimal percent (val>=0<=1.0)
        If it's between 1.0 and 100, it is converted to a decimal percent.
    Args:
        val (float): The value to test / convert to a percentage
        default_val (float): The default value to use in case an invalid amount is passed
    Returns:
        [type]: [description]
    """
    if default_val>1.0 and default_val<=100.0:
        default_val = default_val/100.0
    if default_val>100.0:
        default_val = 0.7

    if val>100:         # The maximum expected is 100%
        return default_val
    elif val>1.0 and val<=100.0:
        val = val/100.00

    return val


# Define a Splitter
def train_test_splitter(X=None, y=None, percent_data=0.7):
    """Manually Splits data into training and testing sets

    Args:
        percent_data (float, optional): The percentage of data to keep. Defaults to 0.7.
    """
    mty_df = pd.DataFrame()
    Nothing = mty_df, mty_df, mty_df, mty_df, False

    if (X is None) or (y is None):
        return Nothing

    percent_data = Pct_Converter(percent_data)
    
    print(f"Type of X is {type(X)}")
    print(f"Type of y is {type(y)}")

    record_count_X = len(X)
    record_count_y = len(y)
    if (record_count_y != record_count_X):
        print("The length of X and y are different")

    split = int(percent_data * record_count_X)

    print(f"X -> Of the {record_count_X} records split=>{split} used to train leaving {record_count_X - split} for testing")
    print(f"y -> Of the {record_count_y} records split=>{split} used to train leaving {record_count_y - split} for testing")



    X_train     = X[:split]
    X_test      = X[split:]
    y_train     = y[:split]
    y_test      = y[split:]

    return X_train, X_test, y_train, y_test, True

class MinMaxOperator:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_test         = X_test
        self.y_train        = y_train
        self.y_test         = y_test
        self.X_train        = X_train

        self.x_train_scaler  = MinMaxScaler()
        self.x_test_scaler   = MinMaxScaler()
        self.y_train_scaler  = MinMaxScaler()
        self.y_test_scaler   = MinMaxScaler()

    def pprint(self):
        for x in [self.X_train, self.x_test, self.y_train, self.y_test]:
            print(f"{x[1]} has shape {x.shape}")

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

def LSTM_Reshaper(df:pd.Series):
    """
    Adds another dimension to the Shape of the DataFrame
    Args:
        df (pd.DataFrame): The DataFrame to reshape
    Returns:
        pd.DataFrame: Reshaped DataFrame with 1 additional dimension.
    """
    print(type(df))

    if df.ndim == 3:
        print("Already Reshaped")
        return df

    rows, cols = df.shape
    if 'reshape' in dir(df):
        return df.reshape((rows, cols, 1))
    else:
        return df

def Generate_Sequential_LSTM_mdl(
    X_train:pd.DataFrame
    , units_count= 30
    , dropout_pct=0.2
    , middle_lyr_cnt = 1
    , compile_option={"optimizer":"adam", "loss":"mean_squared_error"}
    , mdl_name = "My_Seq_Model"
):
    """Generate a LTSM Deep Neural Network Layer Model

    Args:
        X_train (NDFrame): The data to train the model on
        units_count (int, optional): The number of units to use. Defaults to 30.
        dropout_pct (float, optional): The Dropout percentage. Defaults to 0.2.
        middle_lyr_cnt (int, optional): The number of middle layers to create. Defaults to 1.

    Returns:
        Sequential: A sequential Model
    """
    from tensorflow.keras.models import Sequential as LSTMSeqModel
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    # Build the LSTM model. 
    # The return sequences need to be set to True if you are adding additional LSTM layers, but 
    # You don't have to do this for the final layer. 
    # Note: The dropouts help prevent overfitting
    # Note: The input shape is the number of time steps and the number of indicators
    # Note: Batching inputs has a different input shape of Samples/TimeSteps/Features

    # Define the LSTM RNN model.
    model = LSTMSeqModel(name=mdl_name)

    # Initial model setup
    number_units = units_count
    dropout_fraction = Pct_Converter(val=dropout_pct, default_val=0.20)

    # Layer 1
    model.add(LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1))
        )
    model.add(Dropout(dropout_fraction))

    # Layer 2 Middle Layer count
    for _ in range(middle_lyr_cnt):
        model.add(LSTM(units=number_units, return_sequences=True))
        model.add(Dropout(dropout_fraction))

    # Layer 3
    model.add(LSTM(units=number_units))
    model.add(Dropout(dropout_fraction))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(**compile_option)

    return model

def Train_Model(
                model
                , X_train:pd.DataFrame
                , y_train:pd.DataFrame
                , fit_options={"epochs":10, "shuffle":False, "batch_size":20, "verbose":1}):
    
    X_train = LSTM_Reshaper(X_train)

    fit_options["x"]=X_train
    fit_options["y"]=y_train
    

    model.fit(**fit_options)
    
    return model

if __name__ == "__main__":
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    df
    df[['A','B','C']]
    X_train, X_test, y_train, y_test, _ = train_test_splitter(X=df[['A','B','C']], y=df['D'])
    mdl = Generate_Sequential_LSTM_mdl(X_train=X_train)
    mdl = Train_Model(mdl, X_train=X_train, y_train=y_train)
