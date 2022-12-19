#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler

import pymysql
from sqlalchemy import create_engine

# from mySqlConn import 

# %%
df = pd.read_csv(r'data\heart_disease.csv')
def df2tbl(user='', passwd='', dbName='', tblName='', exists=''):
    dbConnStr = f"mysql+pymysql://{user}:{passwd}@localhost/{dbName}"
    dbConn = create_engine(dbConnStr)
    df.to_sql(name=tblName, con=dbConn, if_exists=exists, index=False)

# %%
def tbl2df(userName='', passward='', dbName='', tblName=''):
    conn = pymysql.connect(host='localhost', user=userName, passwd=passward, database=dbName, charset='utf8')
    df = pd.read_sql(f"SELECT * FROM {tblName}", con=conn)
    return df
    
# %%
