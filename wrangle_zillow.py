'''wrangle contains helper functions to assist in data acquisition and preparation
in zillow.ipynb'''
import os,re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from env import get_db_url



def get_zillow_from_sql() -> pd.DataFrame:
    '''
    reads MySQL data from `zillow` database and returns `pandas.DataFrame` with raw data from query
    # Parameters
    None
    # Returns
    parsed DataFrame containing raw data from `zillow` database.
    '''
    query = '''
        SELECT *
    FROM properties_2017
    JOIN predictions_2017 USING(`parcelid`)
    LEFT JOIN airconditioningtype USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    LEFT JOIN storytype USING(storytypeid)
    LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
    WHERE transactiondate < "2018"'''
    return pd.read_sql(query, get_db_url('zillow'))


def df_from_csv(path: str) -> Union[pd.DataFrame, None]:
    '''
    returns zillow DataFrame from .csv if it exists at `path`, otherwise returns None
    # Parameters
    path: string with path to .csv file
    # Returns
    `pd.DataFrame` if file exists at `path`, otherwise returns `None`.
    '''
    if os.path.exists(path):
        return pd.read_csv(path,low_memory=False)

    return None


def wrangle_zillow(from_sql: bool = False, from_csv: bool = False) -> pd.DataFrame:
    '''
    wrangles Zillow data from either a MySQL query or a `.csv` file, prepares the  (if necessary)\
        , and returns a `pandas.DataFrame` object
    containing the prepared Zillow data. If data is acquired from MySQL,
     return `DataFrame` is also encoded to `.csv` in both prepared and
    unprepared states
    ## Parameters
    refresh: if `True`, ignores any `.csv` files and pulls new data from the SQL database,
    default=False.
    ## Return
    parsed and prepared `pandas.DataFrame` with Zillow data from 2017.
    '''
    # aquire Zillow data from .csv if exists
    ret_df = None
    if not from_sql and not from_csv:
        ret_df = df_from_csv('data/prepared_zillow.csv')
        if ret_df is not None:
            return ret_df
    if not from_sql:
        ret_df = df_from_csv('data/zillow.csv')
    if ret_df is None:
        # acquire zillow data from MySQL and caches to data/zillow.csv∏
        ret_df = get_zillow_from_sql()
        ret_df.to_csv('data/zillow.csv', index_label=False)

    return ret_df

def prep_zillow(df:pd.DataFrame)->pd.DataFrame:
    '''
    prepares `DataFrame` for processing
    ## Parameters
    df: `pandas.DataFrame` with unfiltered values    
    ## Returns
    
    '''
    df = df.dropna(subset='logerror')
    df = df.sort_values(by='transactiondate')
    df = df.drop_duplicates(subset=['parcelid'],keep='last')
    cols_to_remove = ['id','id.1']
    for c in df.columns:
        if re.match('.*typeid$',c) is not None:
            cols_to_remove.append(str(c))
    df = df.drop(columns=cols_to_remove)
    return df

def tvt_split(dframe: pd.DataFrame, stratify: Union[str, None] = None,
              tv_split: float = .2, validate_split: float = .3, \
                sample: Union[float, None] = None) -> \
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''tvt_split takes a pandas DataFrame, a string specifying the variable to stratify over,
    as well as 2 floats where 0< f < 1 and
    returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    train_validate, test = train_test_split(
        dframe, test_size=tv_split, random_state=123, stratify=stratify)
    train, validate = train_test_split(
        train_validate, test_size=validate_split, random_state=123, stratify=stratify)
    if sample is not None:
        train = train.sample(frac=sample)
        validate = validate.sample(frac=sample)
        test = test.sample(frac=sample)
    return train, validate, test


def get_scaled_copy(dframe: pd.DataFrame, x: List[str], scaled_data: np.ndarray) -> pd.DataFrame:
    '''copies `df` and returns a DataFrame with `scaled_data`
    ## Parameters
    df: `DataFrame` to be copied and scaled
    x: features in `df` to be scaled
    scaled_data: `np.ndarray` with scaled values
    ## Returns
    a copy of `df` with features replaced with `scaled_data`
    '''
    ret_df = dframe.copy()
    ret_df.loc[:, x] = scaled_data
    return ret_df


def scale_data(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame,
               x: List[str]) ->\
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    scales `train`,`validate`, and `test` data using a `method`
    ## Parameters
    train: `pandas.DataFrame` of training data
    validate: `pandas.DataFrame` of validate data
    test: `pandas.DataFrame` of test data
    x: list of str representing feature columns in the data
    method: `callable` of scaling function (defaults to `sklearn.RobustScaler`)
    ## Returns
    a tuple of scaled copies of train, validate, and test.
    '''
    xtrain = train[x]
    xvalid = validate[x]
    xtest = test[x]
    scaler = RobustScaler()
    scaler.fit(xtrain)
    scale_train = scaler.transform(xtrain)
    scale_valid = scaler.transform(xvalid)
    scale_test = scaler.transform(xtest)
    ret_train = get_scaled_copy(train, x, scale_train)
    ret_valid = get_scaled_copy(validate, x, scale_valid)
    ret_test = get_scaled_copy(test, x, scale_test)
    return ret_train, ret_valid, ret_test
  
def handle_null_cols(df:pd.DataFrame,pct_col:float)-> pd.DataFrame:
    prop_pct = 1-pct_col
    na_sums = pd.DataFrame(df.isna().sum())
    na_sums = na_sums.reset_index().rename(columns={0:'n_nulls'})
    na_sums['percentage'] =  na_sums.n_nulls / df.shape[0]
    ret_indices = na_sums[na_sums.percentage <= prop_pct]['index'].to_list()
    return df[ret_indices]

def handle_null_rows(df:pd.DataFrame, pct_row:float)->pd.DataFrame:
    return df[df.isna().sum(axis=1)/df.shape[1] >= pct_row]


def handle_missing_values(df:pd.DataFrame, pct_row:float,pct_col:float)->pd.DataFrame:
    return handle_null_rows(handle_null_cols(df,pct_col),pct_row)
if __name__ == "__main__":
    df = wrangle_zillow()
    print(df.info())
