#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from drawChart import drawCurve, drawLearningCurve, mlFitPred, drawHisto, drawCorr
from prepro import colsLower, cols2cate

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from mySqlConn import df2tbl, tbl2df

from dataprep.eda import create_report

import warnings
warnings.filterwarnings('ignore')

# %%
# df = pd.read_csv(r'data\heart_disease.csv')

# df2tbl(df, 'root', '******', 'heartdiseasedb', 'heartdiseaseTBL', 'replace')
df = tbl2df('root', '******', 'heartdiseasedb', 'heartdiseaseTBL')

# %%
# report = create_report(df)
# report.save('heart_disease.html')
# %%
print("heart disease df shape :", df.shape)
print('df info :', df.info())

# %%
# 컬럼명 모두 소문자로 변경 / 타겟 컬럼 이름 target으로 변경
df = colsLower(df)

# 타겟값 count 시각화 및 비율
sns.countplot(df, x='target')

print(f"heart disease False :{np.round(df['target'].value_counts()[0] / len(df) * 100, 2)} %")
print(f"heart disease True :{np.round(df['target'].value_counts()[1] / len(df) * 100, 2)} %")

# 연속형 변수인 bmi외 나머지 변수들을 범주형으로 변환
df1 = cols2cate(df)


#%%
# 각각의 범주형 독립변수들간의 관계를 보기 위해 histplot으로 시각화
plt.figure(figsize=(10,15))

drawHisto(df1)

plt.figure(figsize=(8,5))
sns.histplot(data=df1, x='bmi', y='target')

# 컬럼들의 상관관계 시각화
drawCorr(df)

#%%
oriDf = df.copy()

mm_sc = MinMaxScaler()
df.bmi = mm_sc.fit_transform(df[['bmi']])
df
#%%
X = df.drop(['target'], axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
print('X train set shape :', X_train.shape)
print('X test set shape :', X_test.shape)
print('y train set shape :', y_train.shape)
print('y test set shape :', y_test.shape)

#%%
# 모델 정의 (로지스틱, 랜포, xgboost)

lgRg = LogisticRegression(random_state=0)
rfClf = RandomForestClassifier(random_state=0)
xgb = XGBClassifier()

algos = [lgRg, rfClf, xgb]

# 모델 학습 결과 확인 및 roc curve 그리기
plt.figure(figsize=(20,30))
for i, algo in enumerate(algos):
    pred = mlFitPred(algo, X_train, y_train, X_test, y_test)
    drawCurve(y_test, pred, i, algo)
    drawLearningCurve(algo, X_train, y_train, 3, 3, i)
    
#%%
skf = StratifiedKFold(n_splits=40, random_state=None, shuffle=False)
for idx, (tridx, teidx) in enumerate(skf.split(X, y)):
    if idx == 0:
        x_train, x_test = X.iloc[tridx], X.iloc[teidx]
        y_train, y_test = y.iloc[tridx], y.iloc[teidx]
    else:
        pass

#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(x_test)
tdf

#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==1))

#%%
# over sampling 하여 위의 sequence 다시 진행..
Xi = X.drop(['bmi'], axis=1)
Xi = Xi.astype('int')
X = pd.concat([Xi, X.bmi], axis=1)


#%%
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
y_res.value_counts()


# %%
X_trains, X_tests, y_trains, y_tests = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
print("X train set's shape :", X_trains.shape)
print("y train set's shape :", y_trains.shape)
print("X test set's shape :", X_tests.shape)
print("y test set's shape :", y_tests.shape)

# %%
lgRg = LogisticRegression(random_state=0)
rfClf = RandomForestClassifier(random_state=0)
xgb = XGBClassifier()

algos = [lgRg, rfClf, xgb]

# 모델 학습 결과 확인 및 roc curve 그리기
plt.figure(figsize=(20,30))
for i, algo in enumerate(algos):
    pred = mlFitPred(algo, X_trains, y_trains, X_tests, y_tests)
    drawCurve(y_tests, pred, i, algo)
    drawLearningCurve(algo, X_trains, y_trains, 3, 10, i)


#%%
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for idx, (tridx, teidx) in enumerate(skf.split(X_res, y_res)):
    if idx == 0:
        x_train, x_test = X_res.iloc[tridx], X_res.iloc[teidx]
        y_train, y_test = y_res.iloc[tridx], y_res.iloc[teidx]
    else:
        pass

#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(X_train)
tdf

#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==1))

# %%
# under sampling 하여 위의 sequence 다시 진행..
df3 = df.sample(frac=1)
print(df3.target.value_counts())

disDf = df3[df3.target==1.0]
nonDisDf = df3[df.target==0.0][:23893]

print("disease df's sahpe :", disDf.shape)
print("non disease df's sahpe :", nonDisDf.shape)


# %%
newDf = pd.concat([disDf, nonDisDf], axis=0).sample(frac=1)
newDf.shape


# %%
Xu = newDf.drop(['target'], axis=1)
yu = newDf.target

X_trainu, X_testu, y_trainu, y_testu = train_test_split(Xu, yu, test_size=.2, random_state=0)

lgRg = LogisticRegression(random_state=0)
rfClf = RandomForestClassifier(random_state=0)
xgb = XGBClassifier()

algos = [lgRg, rfClf, xgb]

# 모델 학습 결과 확인 및 roc curve 그리기
plt.figure(figsize=(20,30))
for i, algo in enumerate(algos):
    pred = mlFitPred(algo, X_trainu, y_trainu, X_testu, y_testu)
    drawCurve(y_testu, pred, i, algo)
    drawLearningCurve(algo, X_trainu, y_trainu, 3, 10, i)


# %%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(Xu[['stroke', 'highbp', 'diffwalk', 'age', 'genhlth']])
tdf


#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(yu==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(yu==1))
# %%
