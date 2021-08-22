# 03-Aug-21

"""Resources for Classification analysis

# Display the confusion matrix sample usage
n=["High Risk Loans", "Low Risk Loans"]
cm_df = Generate_Confusion_Matrix(y_test=y_test, y_pred=y_pred_brf,names=n)
Describe_Confusion_Matrix(cm_df)

"""



import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def Categorize_Txt_Cols(df,col_name):
    """Categorizes the columns with numbers"""
    le.fit(df[col_name])
    return le.transform(df[col_name])

def Get_Columns_With_TextH(df):
    """Get's alist of text columns from a DataFrame"""
    list_of_columns = list(zip(list(df.keys()), df.dtypes.to_list()))
    return [col[0] for col in list_of_columns if col[1]==np.dtype('O')]

def Convert_Text_Cols(df):
    """Determines which columns are Text"""
    list_of_text_columns = Get_Columns_With_TextH(df)
    
    for col in list_of_text_columns:
        df[col] = Categorize_Txt_Cols(df,col_name=col)
    
    return df


def Combine_Names_for_CF_Matrix(names=None, y_pred=None):
    if names is not None:
        lst = names
    elif y_pred is not None:
        lst = sorted(set(y_pred))

    idx = [f"Actual {x}" for x in lst]
    col = [f"Prediction {x}" for x in lst]

    return idx,col

def Generate_Confusion_Matrix(y_test, y_pred, names=None):
    from sklearn.metrics import confusion_matrix

    idx,col = Combine_Names_for_CF_Matrix(names=names, y_pred=y_pred)

    cm_df = pd.DataFrame(
        data=confusion_matrix(y_test, y_pred)
        , index= idx
        , columns= col
    )
    cm_df['Total Predictions'] = cm_df.sum(axis='columns')
    cm_df.loc["Total Actuals"] = cm_df.sum(axis='index')

    return cm_df
def Calc_Conf_Matrix(cm_df:pd.DataFrame):
    true_negative  = cm_df.iloc[0,0]
    false_positive = cm_df.iloc[0,1]
    false_negative = cm_df.iloc[1,0]
    true_positive  = cm_df.iloc[1,1]
    total_predictions  = cm_df.loc['Total Actuals','Total Predictions']
    total_predict_yes  = cm_df.iloc[2,1]
    total_predict_no     = cm_df.iloc[2,0]
    
    total_actuals_positive = cm_df.iloc[1,2]
    total_actuals_negaitive = cm_df.iloc[0,2]

    accuracy                =  (true_positive  + true_negative ) /total_predictions
    misclassification_rate  =  (false_positive + false_negative) /total_predictions  # Error Rate
    sensitivity_rate        =  true_positive / total_actuals_positive                # True Positive Rate or Recall
    false_positive_rate     =  false_positive / total_actuals_negaitive
    true_negative_rate      =  true_negative  / total_actuals_negaitive              # Specificity
    precision               =  true_positive /  total_actuals_positive

    return {'accuracy':accuracy
            ,'misclassification_rate':misclassification_rate
            ,'sensitivity_rate':sensitivity_rate
            ,'false_positive_rate':false_positive_rate
            ,'true_negative_rate':true_negative_rate
            ,'precision':precision}

def Describe_Confusion_Matrix(cm_df:pd.DataFrame):
    results = Calc_Conf_Matrix(cm_df=cm_df)
    display(cm_df)

    print(f"Overall, how often is the classifier correct (Accuracy): {results['accuracy']:.3%}")
    print(f"Overall, how often is it wrong (Misclassification Rate or Error Rate): {results['misclassification_rate']:.3%}")
    print(f"When it's actually yes, how often does it predict yes? (True Positive Rate or Sensitivity or Recall): {results['sensitivity_rate']:.3%}")
    print(f"When it's actually no, how often does it predict no (False Posive Rate): {results['true_negative_rate']:.3%}")
    print(f"When it predicts yes, how often is it correct? (Precision):{results['precision']:.3%}")
    print(f"When it's actually no, how often does it predict no? (False Positive Rate): {results['false_positive_rate']:.3%}")
    

if __name__ == "__main__":
    pass