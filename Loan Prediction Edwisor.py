#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Analysis¶
# To begin this exploratory analysis, first import libraries and prepare the data.

# In[1]:


# Linear Alzebra
import numpy as np

# Data Processing
import pandas as pd 

# For plotting/Data Visualization
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('dark')

# Algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print('Setup complete')


# ### Loading the data

# In[2]:


df=pd.read_csv('C:\\Users\\bvrav\\OneDrive\\Desktop\\bank-loan.csv')


# In[3]:


df


# ### Verifying the data
# ##### mean
# ##### median
# ##### avg
# ##### std deviation

# In[4]:


df.describe()


# ### Checking the null values

# In[5]:


df.info()


# #### how many persons got loan out of 850

# In[6]:


df['default'].value_counts()


# In[7]:


df.isnull().sum()


# ### correlation

# In[8]:


### lets veify corrolation
df.corr()


# ### A pairs plot allows us to see both distribution of single variables and relationships between two variables

# In[9]:


import seaborn as sns
sns.pairplot(df)


#  
# After going through the dataset in detail and pre-understanding the data the next step is, Methodology
# that will help achieve our goal.
# In Methodology following processes are followed:
# 
# • Pre-processing:
# It includes missing value analysis, outlier analysis, feature selection and feature scaling.
# 
# • Model development:
# It includes identifying suitable Machine learning Algorithms and applying those algorithms in our
# given dataset.

# #### To find out the outliers checking with box plot

# In[10]:


plt.figure(figsize=(20,5))
sns.boxplot(data=df, width=0.3)
plt.show()


# #### verifying with bins what age group got loan

# In[11]:


df['agebyten'] = pd.cut(df['age'],bins=[10,20,30,40,50,60])


# In[12]:


df.groupby(['agebyten']).default.mean()


# ### income group that have loan approved

# In[13]:


df['incomegroup']= pd.cut(df['income'],bins=[0,100,200,300,400,500])


# In[14]:


df['incomegroup']


# In[15]:


df.groupby(['incomegroup']).default.mean()


# #### noticed salary range from 400 to 500 got 100% loan

# In[16]:


df['incomegroup'].value_counts()


# In[17]:


df['ed'].value_counts()


# In[18]:


df.groupby(['ed']).default.mean()


# #### get correlations of each features in dataset

# In[19]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[20]:


# fig, axes = plt.subplots(ncols=len(X.columns), figsize=(30,15))
# for ax, col in zip(axes, X.columns):
#     sns.distplot(X[col], ax=ax)
#     plt.tight_layout() 
# plt.show()


# ###### Performing train, test split for training and testing with different models

# In[21]:


X = pd.DataFrame(df.iloc[:700,:-3])
y=df['default'].iloc[:700]


# ### verifying data distribution, if not we can apply log to make it normal distribution

# In[22]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(30,15))

for i, column in enumerate(X.columns):
    sns.distplot(X[column],ax=axes[i//4,i%4])


# #### checking best feature that effect the default with selectKBest

# In[23]:


from sklearn.feature_selection import SelectKBest, chi2
bestfeatures = SelectKBest(score_func=chi2, k=4).fit_transform(X,y)


# In[24]:


pd.DataFrame(bestfeatures)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2,stratify = y)


# #### Feature scaling is requried to perform certain classifications for accurate results.

# In[26]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[27]:


plt.figure(figsize=(20,5))
sns.boxplot(data=X_train, width=0.3)
plt.show()


# #### Logistic Regression

# In[28]:


model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0)
model.fit(X_train, y_train)


# In[29]:


y_pred=model.predict(X_test)
accuracy_score(y_pred, y_test)


# # SGD Classification and decisiontree models are not required if we are using random forest.

# In[30]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy_score(y_pred, y_test)


# ##### GaussianNB

# In[31]:


gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)  
accuracy_score(y_pred, y_test)


# #### SVM

# In[32]:


linear_svc = LinearSVC(max_iter=10)
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
accuracy_score(y_pred, y_test)


# #### KNN

# In[33]:


knn = KNeighborsClassifier(n_neighbors = 12) 
knn.fit(X_train, y_train)  
y_pred = knn.predict(X_test)  
accuracy_score(y_pred, y_test)


# # from the above models it is obvious that KNN and random forest gives max score
# 
# ## applying hyperparameter tuning for all the selected models 

# ###### For random forest

# In[34]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(X_train, y_train)


# In[35]:


# print best parameter after tuning 
print(gs.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(gs.best_estimator_)


# In[36]:


y_pred = gs.predict(X_test)


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[38]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))


# In[39]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(random_forest, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# In[40]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[41]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[42]:


probs = random_forest.predict_proba(X_test)


# In[43]:


#Keep Probabilities of the positive class only
probs = probs[:, 1]


# In[44]:


auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# In[45]:


fpr, tpr, thresholds = roc_curve(y_test, probs)


# In[46]:


plot_roc_curve(fpr, tpr)


# ### Hyper parameter tuning for KNN model

# In[47]:


knn = KNeighborsClassifier()  
parameters = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.best_estimator_)


# In[48]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=7, p=2,
                     weights='uniform')
knn.fit(X_train, y_train)  
y_pred = knn.predict(X_test)  
accuracy_score(y_pred, y_test)


# In[49]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[50]:


probs = knn.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)


# ### comparing KNN and random forest, AUC values, F1-score are more for the random forest.
# #### so we can apply random forest to the test dataset

# In[51]:


final_test = scaler.fit_transform(df.iloc[701:850,:-3])


# In[52]:


model=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=-1, oob_score=True, random_state=1, verbose=0,
                       warm_start=False)
model.fit(X_train, y_train)
y_pred = model.predict(final_test)


# In[53]:


df2 = pd.Series(y_pred)


# In[54]:


for x,y in enumerate(df2[y],1):
    df['default'].iloc[701:850][x-1:x] = y
    


# In[55]:


df


# In[56]:


df.to_csv("C:\\Users\\bvrav\\OneDrive\\Desktop\\loan prediction update.csv")


# In[57]:


import pickle


# In[59]:


with open('Loan_prediction_pickle','wb') as f:
    pickle.dump(model,f)


# In[60]:


# with open('Loan_prediction_pickle','rb') as f:
# 	model=pickle.load(f)


# In[ ]:




