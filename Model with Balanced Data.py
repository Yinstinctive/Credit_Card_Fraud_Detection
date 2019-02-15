import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\yingk\Desktop\creditcardfraud\creditcard.csv')
df.head()


#0:284315, 1:492, unbalanced data
classes = df['Class'].value_counts()
classes.plot(kind='bar')
plt.show()

#Normalizing Amount Column
df['normAmount'] = StandardScaler().fit_transform(np.array(df['Amount']).reshape(-1,1))
df = df.drop(['Time','Amount'],axis=1)
df.head()

#Train Test Split
X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

def under_sampling(dataframe):
    #Undersampling
    fraud_index = np.array(dataframe[dataframe['Class']==1].index)
    normal_index = np.array(dataframe[dataframe['Class']==0].index)
    random_normal_index = np.random.choice(normal_index,size=len(fraud_index),replace=False)
    
    under_sample_index = np.concatenate([fraud_index, random_normal_index])
    under_sample_df = dataframe.iloc[under_sample_index,:]
    
    X_under_sample = under_sample_df.drop(['Class'],axis=1)
    y_under_sample = under_sample_df['Class']
    
    y_under_sample.value_counts().plot(kind='bar')
    
    return [X_under_sample, y_under_sample]

underSample = under_sampling(df)
X_under_sample = underSample[0]
y_under_sample = underSample[1]

#Logistic Regression
LR = LogisticRegression()
LR.fit(X_under_sample, y_under_sample)

#SVM
svm_model = SVC()
svm_model.fit(X_under_sample, y_under_sample)

#KNN
param_grid = {'n_neighbors':range(1,50)}
grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=1)
grid.fit(X_under_sample, y_under_sample)
print(grid.best_params_)
knn = grid.best_estimator_

#Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_under_sample, y_under_sample)

#Random Forest
param = {'n_estimators':range(1,100)}
grid = GridSearchCV(RandomForestClassifier(),param,refit=True,verbose=1)
grid.fit(X_under_sample, y_under_sample)
print(grid.best_params_)
rfc = grid.best_estimator_

models = [LR, svm_model, knn, dtree, rfc]
predictions = []

def predicting(X_test, model):
    return model.predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_cf_matrix(y_true, y_pred, model):
    cm = confusion_matrix(y_true, y_pred)
    class_names = [0,1]
    plt.figure()
    title = str(type(model)).split('.')[3]
    plot_confusion_matrix(cm, classes=class_names, title = title[:-2])
    plt.show()

for model in models:
    pred = predicting(X_test, model)
    predictions.append(pred)
    get_cf_matrix(y_test, pred, model)

#AUC of Random Forest Model
from sklearn.metrics import classification_report    
rfc_pred = models[-1].predict(X_test)
print(classification_report(y_test, rfc_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,rfc_pred))
