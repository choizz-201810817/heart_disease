#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, roc_curve, auc

def drawHisto(df):
    for i, col in enumerate(df.columns[:-1]):
        plt.subplot(7,3,i+1)
        sns.histplot(data=df.iloc[:,:-1], x=col, y='target')
    plt.tight_layout()
    plt.show()

def drawCorr(df):
    mask = np.zeros_like(df.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)]=True

    plt.figure(figsize=(12,10))
    sns.heatmap(data=df.corr(), mask=mask, cmap='coolwarm_r', linewidths=1)
    print(df.corr().target.sort_values())

def mlFitPred(algo, X_train, y_train, X_test, y_test):
    algo.fit(X_train, y_train)
    pred = algo.predict(X_test)
    print(f"<<< {algo.__class__.__name__}'s classification report >>>")
    print(classification_report(y_test, pred))
    return pred

def drawCurve(y_test, pred, i, algo):
    fpr, tpr, th = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    print(f'{algo.__class__.__name__} roc_auc score :', roc_auc)

    plt.subplot(3,2,(((i+1)*2)-1))
    plt.title(f'{algo.__class__.__name__} roc curve')
    plt.plot(fpr, tpr, color='red', label="AUC : {0:.02f}".format(roc_auc))
    plt.plot([0,1],[0,1],linestyle='--')

def drawLearningCurve(algo, X, y, cv, epochs, i):
    trainSize, trainScore, testScore = learning_curve(algo, X, y, cv=cv, n_jobs=1, train_sizes=np.linspace(.1, 1.0, epochs))
    trainScoreMean = np.mean(trainScore, axis=1)
    testScoreMean = np.mean(testScore, axis=1)
    plt.subplot(3,2,(i+1)*2)
    plt.title(f'{algo.__class__.__name__} learning curve')
    plt.plot(trainSize, trainScoreMean, '-o', color='blue', label='Train Score')
    plt.plot(trainSize, testScoreMean, '-o', color='red', label='Test Score')
    plt.legend(loc = 'best')
# %%
