import pandas as pd
import numpy as np

# 컬럼명 모두 소문자로 변경 / 타겟 컬럼 이름 target으로 변경
def colsLower(df):
    df.columns = df.columns.str.lower()
    df.rename(columns={'heartdiseaseorattack':'target'}, inplace=True)
    return df

# 연속형 변수인 bmi외 나머지 변수들을 범주형으로 변환
def cols2cate(df):
    df1 = df.drop(['bmi'],axis=1)
    df1 = df1.astype('int').astype('category')
    df2 = pd.concat([df1, df['bmi']], axis=1)
    return df2