#%%
from sqlalchemy import create_engine
import pymysql
import pandas as pd

def df2tbl(df, user='', passwd='', dbName='', tblName='', exists=''):
    dbConnStr = f"mysql+pymysql://{user}:{passwd}@localhost/{dbName}"
    dbConn = create_engine(dbConnStr)
    df.to_sql(name=tblName, con=dbConn, if_exists=exists, index=False)

def tbl2df(userName='', passward='', dbName='', tblName=''):
    conn = pymysql.connect(host='localhost', user=userName, passwd=passward, database=dbName, charset='utf8')
    df = pd.read_sql(f"SELECT * FROM {tblName}", con=conn)
    return df
# %%
