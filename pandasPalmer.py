from calendar import month
import pandas as pd
import numpy as np
import os.path as path
import datetime
from datetime import datetime
from pathlib import Path, PurePath

# try:
#     from urllib.parse import urlparse
#     from urllib.request import request
# except ImportError:
#     try:
#         from urlparse import urlparse
#         from urlrequest import request
#     except ImportError:
#         print('Sorry ran out of options')

import urllib.parse as urlparse
import urllib.request as request, json

def import_func(file_or_url):
    """Private function to use the file extension to determine 
    the Pandas import function to use"""
    # Locate the text at the last period '.'
    _, extn = path.splitext(file_or_url.lower())
    extn_fns = {  '.csv':pd.read_csv
                , '.xls':pd.read_excel
                , '.xlsx':pd.read_excel
                , '.json':pd.read_json
                , '.html':pd.read_html
            }
    return extn_fns[extn]

def is_valid_url(url):
    """Determines if the URL is valid or not."""
    rslt = urlparse.urlparse(url)
    is_url = all([rslt.scheme, rslt.netloc, rslt.path])
    
    is_url_valid = False
    
    if is_url:
        try:
            with request.urlopen(url) as resp:
                is_url_valid = (resp.status == 200)
        except Exception as exn:
            is_url_valid = False
            print(f'There was an error {exn}')
        else:
            is_url_valid = True
            
    return is_url_valid

def Get_json(json_path:str):
    if is_valid_url(json_path):
        with request.urlopen(json_path) as url:
            data = json.loads(url.read().decode())

            if data["Response"]=="Success":
                all_data = data
            else:
                print("NOT ABLE TO READ THE DATA!!!")
                all_data = None
    elif(path.exists(json_path)):
        with open(json_path) as f:
            data = json.load(f)

            if data is not None:
                all_data = data
            else:
                all_data = None
    return all_data

def apply_index(df, idx=None):
    """Apply an index to a dataframe"""
    right_type = type(idx) == list or type(idx) == str
    idx_in_df = set(idx).issubset(set(df))
    not_already_idxd = not (set(idx).issubset(df.index.names) or idx in df.index.names)

    if right_type and idx_in_df and not_already_idxd:
        df.set_index(keys=idx,inplace=True)
    else:
        # TODO: Fix this later, check if the index already exists
        #print(f"Index not applied {idx} [Type {right_type}] [Subset {idx_in_df}]")
        pass

    return df

def CreateDataFrame(file_name, idx=None, remove_nulls=True):
    """Creates a DataFrame the way I like it."""
    if type(file_name) == str:
        file_name = file_name.strip()
    elif file_name.issubclass(PurePath):
        pass

    state = []

    apply_func = import_func(file_name)
    if is_valid_url(file_name):
        state.append('Valid url')
    else:
        if not path.exists(file_name):
            print(f'Had problems locating the data [{file_name}]')
            return pd.DataFrame()

    # Import File based on file extension
    df = apply_func(file_name, index_col=idx, parse_dates=True, infer_datetime_format=True)
    state.append('imported')

    if remove_nulls:
        df.dropna(inplace=True)

    print(state)

    return df

# Check if the object is a list or dictionary
Is_Iterable = lambda objX: ('__getitem__' in dir(objX) or '__iter__' in dir(objX))
# Split up the componets of a list of tuples
unzip_list  = lambda zlst: list(zip(*zlst))


# Get the Quarter, based on the date
Qtr = lambda dtx: (dtx.month+2)//3
# Get the Last Month of the Qtr
Qtr_EOM = lambda dtx: Qtr(dtx) * 3

# Let's put it all together
def Last_Day_Qtr(dtx:datetime.date):
    """Get the last day of the month in a Given Qtr"""
    Qtr_Last_Day_LU = {1:31,2:30,3:30,4:31}

    yr = dtx.year
    m  = Qtr_EOM(dtx)
    d  = Qtr_Last_Day_LU[ Qtr(dtx) ]
    return datetime(yr,m,d)

# Get the number of days from the End of Qtr to given date
Days_2_EoQ = lambda dtx: (Last_Day_Qtr(dtx) - dtx).days

def Get_Qtr(xdate:datetime):
    """ This function takes various datetime objects and returns the Quarter associated with the values passed in
        So for an array, it will return an array of quarters as string.  If a single value was passed the
        a single quarter will be returned."""
    from calendar import monthrange
    import numpy as np
    typ = type(xdate)
    if typ is datetime or typ is datetime.date:
        q = (xdate.month+2)//3
        y = xdate.year
        return (f"{y}Q{q}")
    elif typ is np.ndarray:
        lst_date = [Get_Qtr(x) for x in xdate]
        return lst_date
    elif typ is pd.Series or typ is list:
        lst_date = [Get_Qtr(x[1]) for x in xdate.iteritems()]
        return lst_date
    elif typ is pd.core.indexes.multi.MultiIndex:
        lst_date = [Get_Qtr(x[1].date()) for x in xdate.values]
        return lst_date
    elif typ is pd.core.indexes.datetimes.DatetimeIndex:
        lst_date = [Get_Qtr(x.date()) for x in xdate]
        return lst_date
    elif typ is str:
        return Get_Qtr(datetime.strptime(xdate, '%Y-%m-%d'))
    else:
        #print(f'The type is {type(xdate)} not planned for.')
        d = xdate
        q = (d.month+2)//3
        y = d.year
        return (f"{y}Q{q}")
def Get_Date(xdate:datetime=None):
    """ This function takes various datetime objects and returns the Date associated with the values passed in.
        So for an array, it will return an array of datetime dates.  If a single value was passed the
        a single date will be returned."""
    if xdate is None:
        return None

    if 'mro' in (dir(xdate)):
        print(f'{xdate.mro()}')
    
    import numpy as np
    import pandas as pd

    typ = type(xdate)
    
    if typ is datetime or typ is datetime.date:
        return datetime(xdate.year,xdate.month,xdate.day)
    elif (typ is np.ndarray):
        lst_date = [Get_Date(x) for x in xdate]
        return lst_date
    elif (typ is pd._libs.tslibs.timestamps.Timestamp) and Is_Iterable(xdate):
        #print('pd._libs.tslibs.timestamps.Timestamp ITERABLE')
        return [Get_Date(x) for x in xdate]
    elif (typ is pd._libs.tslibs.timestamps.Timestamp):
        #print('pd._libs.tslibs.timestamps.Timestamp')
        return pd.to_datetime(xdate).date() 
    elif typ is pd.Series or typ is list:
        lst_date = [Get_Date(x[1]) for x in xdate.iteritems()]
        return lst_date
    elif typ is pd.core.indexes.multi.MultiIndex:
        lst_date = [Get_Date(x[1].date()) for x in xdate.values]
        return lst_date
    elif typ is pd.core.indexes.datetimes.DatetimeIndex:
        lst_date = [Get_Date(x.date()) for x in xdate]
        return lst_date
    elif typ is str:
        return datetime.strptime(xdate, '%Y-%m-%d')
    else:
        # print(type(xdate))
        return xdate
        # return datetime(xdate.year, xdate.month, xdate.day)
def Linker(sym:pd.Series, dtx:pd.Series):
    """Takes a Stock Symbol as a string and a date and returns a string in the format ABCYYYMMDD"""
    if Is_Iterable(sym) and Is_Iterable(dtx) and len(sym)==len(dtx):
        # pd.to_datetime(pp.Get_Date(df_bal_sheet.index[0])).strftime('%Y%m%d')
        dates = [pd.to_datetime(Get_Date(x)).strftime('%Y%m%d') for x in dtx]
        jnd = zip(sym,dates)
        return [x[0]+x[1] for x in jnd]
    else:
        return None

def ML_Imports():
    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
    except ImportError as ie:
        print(f'Sorry failed to import the sci-kit learn modules: {ie.msg}')
def Palmer_Import():
    import importlib.util
    import sys
    # import tokenize
    # file_path = tokenize.__file__
    # modle_name = tokenize.__name__

    spec = importlib.util.spec_from_file_location("pandasPalmer", "D:\\Data\\Family\\Ethol Palmer\\UoT SCS\\FinTech\Repos\\lib\\pandasPalmer.py")
    pandasPalmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pandasPalmer)
    return None

# import seaborn as sns
# titanic = sns.load_dataset('titanic')
def Get_Target_and_Features(df:pd.DataFrame, target_name='Outcome'):
    if not set(df.columns).issubset(target_name):
        print(f"Sorry the column '{target_name}' is not a column in {list(df.columns)}")
        return (None,None)
    if len(df.columns)<2:
        print("Sorry not enough colums to perform operation")
        return (None,df[target_name])
    return (df.drop(columns=target_name), df[target_name])

# 03-Aug-21
def Get_Columns_With_Text(df):
    """Get's a list of text columns from a DataFrame"""
    list_of_columns = list(zip(list(df.keys()), df.dtypes.to_list()))
    return [col[0] for col in list_of_columns if col[1]==np.dtype('O')]



if __name__ == "__main__":
    # execute only if run as a script
    # if is_valid_url('https://www.youtube.com/watch?v=ucY6NwQTI3M&list=RDMMnQWFzMvCfLE&index=9'):
    #     print(urlparse.urlparse('https://www.youtube.com/watch?v=ucY6NwQTI3M&list=RDMMnQWFzMvCfLE&index=9'))
    #main()
    # dtx_lst = [datetime.date(2011, 1, 4), datetime.date(2011, 1, 5), datetime.date(2011, 1, 6), datetime.date(2011, 1, 7), datetime.date(2011, 1, 10),
    #             datetime.date(2011, 1, 11), datetime.date(2011, 1, 12), datetime.date(2011, 1, 13), datetime.date(2011, 1, 14), datetime.date(2011, 1, 18)]
    # results = map(Days_2_EoQ,dtx_lst)

    # print(Days_2_EoQ(datetime(2011, 1, 12)))
    # print(datetime.now())
    # print(Qtr(datetime.now()))
    # print(datetime.now() + pd.tseries.offsets.MonthEnd(2))
    # print(Get_Date())
    # print(Get_Date(datetime(2012,12,2)))
    # print(Get_Qtr('2012-12-2'),Get_Qtr(datetime(2012,12,2)),Get_Date('2019-10-30'),Linker('AAL','2012-11-20'))
    # Palmer_Import()
    # ML_Imports()
    # d = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pd.DataFrame(data=d)
    # X,y = Get_Target_and_Features(df)
    # print(f"X={X} and y={y}")

    url_site = "https://min-api.cryptocompare.com/data/all/coinlist"
    x = Get_json(url_site)
    print(x)