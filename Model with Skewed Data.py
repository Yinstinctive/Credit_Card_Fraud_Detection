import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\yingk\Desktop\creditcardfraud\creditcard.csv')
df.head()
sns.heatmap(df.isnull(), yticklabels = False, cbar=False, cmap='viridis')
plt.show()
df.isna().sum()

corr = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, cmap='coolwarm',linewidths=0.5)
plt.show()

features = df.drop(['Class'],axis=1)
target = df['Class']

#StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)
df_scaled_features = pd.DataFrame(data = scaled_features, columns = features.columns)
df_scaled_features.head()

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_scaled_features, target, test_size=0.3)

#KNN
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

#Decide value of n_neighbors
# =============================================================================
# for i in range(1,20):
#     knn = KNeighborsClassifier(n_neighbors = i)
#     knn.fit(X_train,y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i!=y_test))
#     print(f'k={i} completed')
# plt.figure(figsize=(10,6))
# plt.plot(range(1,20), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# =============================================================================

# n_neighbors = 18
knn = KNeighborsClassifier(n_neighbors = 18)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# GrudSearchCV
# =============================================================================
# param_grid={'n_neighbors':range(1,100)}
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=3)
# grid.fit(X_train,y_train)
# =============================================================================

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)

print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

df['Class'].value_counts()
#class 1/total = 0.001727

#SVM
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))