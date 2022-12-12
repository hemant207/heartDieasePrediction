#!/usr/bin/env python
# coding: utf-8

# # importing necessery librairies 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # read data

# In[2]:


from scipy.io import arff


# In[3]:


df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
df


# In[4]:


df.columns


# In[5]:


df.info()


# we check data we imported and we can see there are 12 total columns with 1190 nonnull numeric values.target is our dependent variable.

# # check data 

# In[6]:


df.head()


# # now we will check fro the data shape and data description.

# In[7]:



df.describe()


# In[8]:


df.shape


# In[9]:


cat_df = ["sex","chest pain type"]
num_df = []

for x in df.columns:
    if x not in cat_df:
        num_df.append(x)
    else:
        pass
        


# In[10]:


num_df


# In[11]:


#now we we will plot data for data exploeration
#first we will plot categorical data and then we plot numerical data


# In[12]:


sns.displot(data=df,x="sex",hue='target')


# In[13]:


sns.displot(data=df,x="chest pain type",hue='target')


# In[14]:


sns.catplot(data=df, kind="bar", x="chest pain type", y="sex", hue="target")


# In[15]:


corr = df.corr()
corr


# In[69]:


plt.figure(figsize=(10,10))
sns.heatmap(corr,annot=True,cmap="Blues")


# In[70]:


plt.figure(figsize=(15,15))
pair_p = sns.pairplot(data=df)


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x = df[['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
       'fasting blood sugar', 'resting ecg', 'max heart rate',
       'exercise angina', 'oldpeak', 'ST slope']]
y = df['target']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# # knn classifier

# In[21]:


from sklearn.neighbors import KNeighborsClassifier


# In[22]:


knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
knn_model.fit(X_train, y_train)  


# In[23]:


y_pred = knn_model.predict(X_test)
y_pred


# In[24]:


from sklearn.metrics import confusion_matrix
confused = confusion_matrix(y_test, y_pred)
confused


# In[25]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
print(accuracy_score(y_test, y_pred))


# In[26]:


test_error_rates = []


for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train) 
   
    y_pred = knn_model.predict(X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred)
    test_error_rates.append(test_error)


# In[27]:


plt.figure(figsize=(6,4),dpi=100)
plt.plot(range(1,30),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")


# In[28]:


knn_model_2 = KNeighborsClassifier(n_neighbors=11, metric='minkowski', p=2 )
knn_model_2.fit(X_train, y_train)


# In[29]:


y_pred_2 = knn_model_2.predict(X_test)
y_pred_2


# In[30]:


confused = confusion_matrix(y_test, y_pred)
confused


# # decision tree as a classifier

# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


df


# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[73]:


tree_model = DecisionTreeClassifier()


# In[74]:


tree_model.fit(X_train,y_train)


# In[75]:


from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(tree_model,filled=True)


# In[37]:


y_pred = tree_model.predict(X_test)


# In[38]:


tree_acc_1 = accuracy_score(y_pred,y_test)
tree_acc_1


# In[39]:


tree_report_1 = classification_report(y_pred,y_test)
print(tree_report_1)


# In[40]:


para = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best', 'random'],
    'max_depth' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    
}


# In[41]:


from sklearn.model_selection import GridSearchCV


# In[42]:


tree_model = DecisionTreeClassifier()
cv = GridSearchCV(tree_model,param_grid=para,cv=7,scoring='accuracy')


# In[43]:


cv.fit(X_train,y_train)


# In[44]:


cv.best_params_


# In[45]:


tree_pred = cv.predict(X_test)


# In[46]:


tree_acc_2 = accuracy_score(tree_pred,y_test)
tree_acc_2


# In[47]:


tree_report_2 = classification_report(tree_pred,y_test)
print(tree_report_2)


# In[48]:


print(tree_report_1)


# # naive bayes alogorithm

# In[49]:


from sklearn.naive_bayes import GaussianNB


# In[50]:


gnb_model = GaussianNB()


# In[51]:


gnb_model.fit(X_train,y_train)


# In[52]:


y_pred_gnb = gnb_model.predict(X_test)


# In[53]:


acc_gnb = accuracy_score(y_pred_gnb,y_test)
acc_gnb


# In[54]:


gnb_report = classification_report(y_pred_gnb,y_test)
print(gnb_report)


# # random forest classifier

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


rf_model = RandomForestClassifier()


# In[57]:


rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)


# In[58]:


acc_rf = accuracy_score(y_pred_rf,y_test)
acc_rf


# In[59]:


rf_report = classification_report(y_pred_rf,y_test)
print(rf_report)


# In[60]:


rf_para= {
    'n_estimators' : [100,200,300,400,500,600,700],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'oob_score' : [True,False],
}


# In[61]:


cv_rf = GridSearchCV(rf_model,param_grid=rf_para,cv=7,scoring='accuracy')


# In[62]:


cv_rf.fit(X_train,y_train)


# In[63]:


cv_rf.best_params_


# In[64]:


y_pred_rf_2 = cv_rf.predict(X_test)


# In[65]:


acc_rf_2 = accuracy_score(y_pred_rf_2,y_test)
acc_rf_2
  


# In[66]:


rf_report_2 = classification_report(y_pred_rf_2,y_test)
print(rf_report_2)


# In[67]:


print(rf_report)


# # pickling model for deployment usage

# In[77]:


import pickle


# In[81]:


pickle.dump(cv_rf,open('random_forest.pkl','wb'))


# In[88]:


r_model = pickle.load(open('random_forest.pkl','rb'))


# In[90]:


y1 = r_model.predict(X_test)


# In[91]:


accuracy_score(y1,y_test)

