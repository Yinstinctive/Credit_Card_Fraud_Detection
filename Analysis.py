import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\yingk\Desktop\creditcardfraud\creditcard.csv')
df.head()
df.describe()

features = df.drop(['Class'],axis=1)
target = df['Class']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=101)

#LogisticRegression Benchmark=0.798
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
print(roc_auc_score(y_test,log_pred))

# Under Sampling
def under_sampling(dataframe):
    #Undersampling
    number_records_fraud = len(dataframe[dataframe['Class'] == 1])
    fraud_index = np.array(dataframe[dataframe['Class']==1].index)
    normal_index = dataframe[dataframe['Class']==0].index
    random_normal_index = np.random.choice(normal_index,number_records_fraud,replace=False)
    random_normal_index = np.array(random_normal_index)
    
    under_sample_index = np.concatenate([fraud_index, random_normal_index])
    under_sample_df = dataframe.iloc[under_sample_index,:]
    
    X_under_sample = under_sample_df.drop(['Class'],axis=1)
    y_under_sample = under_sample_df['Class']
    
    y_under_sample.value_counts().plot(kind='bar')
    print(y_under_sample.value_counts())
    
    return [X_under_sample, y_under_sample]

df_train = X_train
df_train['Class'] = y_train
df_train.reset_index(drop=True, inplace=True)
underSample = under_sampling(df_train)
X_under_sample = underSample[0]
y_under_sample = underSample[1]

#Z-Normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(X_under_sample)

# =============================================================================
# #PCA
# df_ft_scaled = pd.DataFrame(features_scaled, columns=features.columns)
# corr = df_ft_scaled.corr()
# plt.figure(figsize=(12,12))
# sns.heatmap(corr, cmap='coolwarm',linewidths=0.5)
# plt.show()
# 
# pca = PCA(n_components = 0.99)
# ft_scaled_pca = pca.fit_transform(df_ft_scaled)
# print(pca.explained_variance_ratio_.sum())
# print(ft_scaled_pca.shape)
# pca_ft_num = ft_scaled_pca.shape[1]
# df_pca = pd.DataFrame(ft_scaled_pca, columns=[f'V{i}' for i in range(1,pca_ft_num+1)])
# df_pca.head()
# =============================================================================

# =============================================================================
# #Outliers
# outliers_rows, outliers_columns = np.where(np.abs(features_scaled.values)>3.0)
# print(outliers_rows.shape)
# =============================================================================

#Cross Validation
lr = LogisticRegression()
svm_model = SVC()
dtree = DecisionTreeClassifier()
for model in [lr, svm_model, dtree]:
    scores = cross_val_score(model, features_scaled, y_under_sample, cv=10, scoring='roc_auc')
    print(f'{type(model)}, Mean:{np.mean(scores)}, Std: {np.std(scores)}')
#Champion model: SVC, F1 mean:0.9336, std:0.02676

#GridSearch
svc_search = SVC()
search_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel':['linear','poly','rbf','sigmoid']}
search_func = GridSearchCV(estimator = svc_search, param_grid = search_grid, scoring = 'roc_auc', iid=True, refit=True, cv=10)
search_func.fit(features_scaled, y_under_sample)
print(search_func.best_params_)
print(search_func.best_score_)
svc_improved = search_func.best_estimator_
# 'C':10, 'kernel':'poly'
# best_score_: 0.9367

#Random Forest
param = {'n_estimators':range(1,100)}
grid = GridSearchCV(RandomForestClassifier(),param,scoring = 'roc_auc', iid=True, refit=True, cv=10)
grid.fit(features_scaled, y_under_sample)
print(grid.best_params_)
print(grid.best_score_)
#'n_estimator': 96
# best_score: 0.9793
rfc = grid.best_estimator_

scaler2 = StandardScaler()
features_scaled2 = scaler.fit_transform(X_test)

svc_improved_pred = svc_improved.predict(features_scaled2)
rfc_pred = rfc.predict(features_scaled2)
print("--------------------------------------------------")
print('Improved SVC')
print(confusion_matrix(y_test, svc_improved_pred))
print(classification_report(y_test, svc_improved_pred))
print(roc_auc_score(y_test,svc_improved_pred))
print("--------------------------------------------------")
print('Random Forest')
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
print(roc_auc_score(y_test,rfc_pred))
