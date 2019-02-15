```Python
df = pd.read_csv(r'C:\Users\yingk\Desktop\creditcardfraud\creditcard.csv')
df.head()

classes = df['Class'].value_counts()
classes.plot(kind='bar')
plt.show()
```
![class_count](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/classes_count.png)<br>
There are 284315 records under Class 0 and only 492 records under Class 1, the original data is clearly **unbalanced**.<br>
If proceed with this unbalanced data, we could just use the majority class to assign to all records, the accuracy would still be high. But we will classify all "1" incorrectly. In our case, minimize the misclassification of "1"(fraud) should be our first priority. Next we will appy logistic regression to both skewed data and balanced data, and compare the model performance.<br>
**Logistic Regression with Skewed Data**<br>
```Python
#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_scaled_features, target, test_size=0.3)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)

print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
```
![LR-with skewed data](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/LR-with skewed data.png)<br>
The model misclassified 48 frauds. Let's check out if it could get better after balancing the data.<br>

**Under Sampling**<br>
```Python
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
```
![under_sample_data](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/under_sample_data.png)<br>
Class 0 and Class 1 have reached the 50:50 ratio.<br>

**Apply different models using balanced data**<br>
```Python
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
```
**Plot the confusion matrix and compare model performance**<br>
```Python
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
```
![LR](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/LR.png)<br>
Compare the new LR model with the previous one, the missclassification of "1" has decreased to only 8 records. But precision dropped as well.<br> 
![SVM](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/SVM.png)<br>
![KNN](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/KNN.png)<br>
![DT](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/DT.png)<br>
![RF](https://github.com/Yinstinctive/Credit_Card_Fraud_Detection/blob/master/image/RF.png)<br>
It turns out the random forest model has the best performance among these models. Recall=1 while keep a comparatively high precision.<br>
