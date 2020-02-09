#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
os.chdir("C:\\Users\\antony.morais\\Desktop\\Amalraj\\Titanic")
os.getcwd()


# In[3]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#train.head(5)


# In[4]:


train["Survived"] = train["Survived"].astype('category')
train["Embarked"] = train["Embarked"].astype('category')
test["Embarked"] = test["Embarked"].astype('category')
train["Sex"] = train["Sex"].astype('category')
test["Sex"] = test["Sex"].astype('category')


# In[5]:


from pandas.api.types import CategoricalDtype
pclass=train['Pclass'].unique().tolist()
pclass.sort()
pclass_type = CategoricalDtype(categories=pclass,ordered=True)
train['Pclass'] = train['Pclass'].astype(pclass_type)
test['Pclass'] = test['Pclass'].astype(pclass_type)


# In[6]:


sibSp=train['SibSp'].unique().tolist()
sibSp.sort()
sibSp_type = CategoricalDtype(categories=sibSp,ordered=True)
train["SibSp"] = train["SibSp"].astype(sibSp_type)
test["SibSp"] = test["SibSp"].astype(sibSp_type)


# In[7]:


parch=train['Parch'].unique().tolist()
parch.sort()
parch_type = CategoricalDtype(categories=parch,ordered=True)
train["Parch"] = train["Parch"].astype(parch_type)
#There is a problem with the no of categories in parch in test data. It has one more category.
parch=test['Parch'].unique().tolist()
parch.sort()
parch_type = CategoricalDtype(categories=parch,ordered=True)
test["Parch"] = test["Parch"].astype(parch_type)


# In[8]:


train_sub = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_sub=test.drop(['Name','Ticket','Cabin'],axis=1)


# In[9]:


train_sub.count()


# In[10]:


train_sub.isnull().sum()


# In[11]:


def add_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(train_sub[train_sub["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age
train_sub["Age"] = train_sub[["Age", "Pclass"]].apply(add_age,axis=1)


# In[12]:


def add_aget(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(test_sub[test_sub["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age
test_sub["Age"] = test_sub[["Age", "Pclass"]].apply(add_aget,axis=1)


# In[13]:


def mod_emb(cols):
    Embarked = cols[0]
    if pd.isnull(Embarked):
        return 'S'
    else:
        return Embarked
train_sub["Embarked"] = train_sub[["Embarked"]].apply(mod_emb,axis=1)
test_sub["Embarked"] = test_sub[["Embarked"]].apply(mod_emb,axis=1)
#train_sub.Embarked.dtype
train_sub["Embarked"] = train_sub["Embarked"].astype('category')
test_sub["Embarked"] = test_sub["Embarked"].astype('category')


# In[14]:


one_hot = pd.get_dummies(train_sub['Sex'])
train_sub = train_sub.join(one_hot)
train_sub = train_sub.drop('Sex',axis = 1)
one_hot = pd.get_dummies(train_sub['Embarked'])
train_sub = train_sub.drop('Embarked',axis = 1)
train_sub = train_sub.join(one_hot)
one_hot = pd.get_dummies(train_sub['Pclass'])
one_hot.columns = ["PC1","PC2","PC3"]
train_sub = train_sub.drop('Pclass',axis = 1)
train_sub = train_sub.join(one_hot)


# In[15]:


one_hot = pd.get_dummies(test_sub['Sex'])
test_sub = test_sub.join(one_hot)
test_sub = test_sub.drop('Sex',axis = 1)
one_hot = pd.get_dummies(test_sub['Embarked'])
test_sub = test_sub.drop('Embarked',axis = 1)
test_sub = test_sub.join(one_hot)
one_hot = pd.get_dummies(test_sub['Pclass'])
one_hot.columns = ["PC1","PC2","PC3"]
test_sub = test_sub.drop('Pclass',axis = 1)
test_sub = test_sub.join(one_hot)


# In[16]:


train_sub["male"] = train_sub["male"].astype('category')
train_sub["female"] = train_sub["female"].astype('category')
train_sub["C"] = train_sub["C"].astype('category')
train_sub["Q"] = train_sub["Q"].astype('category')
train_sub["S"] = train_sub["S"].astype('category')
train_sub["PC1"] = train_sub["PC1"].astype('category')
train_sub["PC2"] = train_sub["PC2"].astype('category')
train_sub["PC3"] = train_sub["PC3"].astype('category')


# In[17]:


test_sub["male"] = test_sub["male"].astype('category')
test_sub["female"] = test_sub["female"].astype('category')
test_sub["C"] = test_sub["C"].astype('category')
test_sub["Q"] = test_sub["Q"].astype('category')
test_sub["S"] = test_sub["S"].astype('category')
test_sub["PC1"] = test_sub["PC1"].astype('category')
test_sub["PC2"] = test_sub["PC2"].astype('category')
test_sub["PC3"] = test_sub["PC3"].astype('category')


# In[18]:


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
high_var=sel.fit_transform(train_sub)


# In[19]:


#Tells the no of columns worth out of the total number of columns
print(len(train_sub.columns[sel.get_support()]))
useless = [column for column in train_sub.columns
                    if column not in train_sub.columns[sel.get_support()]]
print(useless)
train_sub = train_sub.drop(['C','Q'],axis=1)
test_sub = test_sub.drop(['C','Q'],axis=1)
#train_sub.nunique()


# In[20]:


train_sub['male_cnt'] = train_sub['male'].astype('int64')
target_enc_m=train_sub.loc[train_sub.Survived == 1, "male_cnt"].sum()
train_sub['female_cnt'] = train_sub['female'].astype('int64')
target_enc_f=train_sub.loc[train_sub.Survived == 1, "female_cnt"].sum()


# In[21]:


tar_en_overall = target_enc_m + target_enc_f
train_sub['sex_target_encoded'] = np.where(train_sub['female']==1, target_enc_f/tar_en_overall,target_enc_m/tar_en_overall)
#Target Encoding of train data will be used for test data as there is no 'Survived' column
tar_en_overall = target_enc_m + target_enc_f
test_sub['sex_target_encoded'] = np.where(test_sub['female']==1, target_enc_f/tar_en_overall,target_enc_m/tar_en_overall)


# In[22]:


#train_sub['sex_target_encoded']


# In[23]:


del train_sub['male_cnt']
del train_sub['female_cnt']


# In[24]:


#!pip install woe
#import woe


# In[25]:


#!pip install MonotonicBinning
#import MonotonicBinning


# In[26]:


#pd.qcut(train_sub['Age'], q=6)
#0.419, 19.0, 25.0, 26.0, 32.5, 40.5, 80.0


# In[27]:


#pd.qcut(train_sub['Fare'], q=3)
#(-0.001, 8.662] < (8.662, 26.0] < (26.0, 512.329]


# In[28]:


bins = [-0.001, 8.662, 26.0, 512.329]
labels = ['Low_Fare','Medium_Fare','High_Fare']
train_sub['fare_binned'] = pd.cut(train_sub['Fare'], bins=bins,labels=labels)
#print (train_sub['fare_binned'].value_counts())
test_sub['fare_binned'] = pd.cut(test_sub['Fare'], bins=bins,labels=labels)


# In[29]:


#bins = [0, 3, 12, 18, 22, 25, 29, 33, 40, 50, 100]
bins = [0.419, 19.0, 25.0, 26.0, 32.5, 40.5, 80.0]
labels = ['age_1','age_2','age_3','age_4','age_5','age_6']
train_sub['age_binned'] = pd.cut(train_sub['Age'], bins=bins,labels=labels)
print (train_sub['age_binned'].value_counts())


# In[30]:


test_sub['age_binned'] = pd.cut(test_sub['Age'], bins=bins,labels=labels)


# In[31]:


pd.pivot_table(test_sub,index='Parch',columns='fare_binned',values='male',aggfunc=sum,fill_value=0)


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.pivot_table(test_sub,index='Parch',columns='fare_binned',values='male',aggfunc=sum,fill_value=0).plot()


# In[33]:


#qcut :
#The pandas documentation describes qcut as a “Quantile-based discretization function.” 
#This basically means that qcut tries to divide up the underlying data into equal sized bins. 
#The function defines the bins using percentiles based on the distribution of the data, and not the actual numeric edges of the bins.


# In[34]:


#train_sub['Age'].describe()


# In[35]:


del train_sub['Age']
del train_sub['Fare']
one_hot = pd.get_dummies(train_sub['fare_binned'])
train_sub = train_sub.join(one_hot)
train_sub = train_sub.drop('fare_binned',axis = 1)
one_hot = pd.get_dummies(train_sub['age_binned'])
train_sub = train_sub.join(one_hot)
train_sub = train_sub.drop('age_binned',axis = 1)


# In[36]:


del test_sub['Age']
del test_sub['Fare']
one_hot = pd.get_dummies(test_sub['fare_binned'])
test_sub = test_sub.join(one_hot)
test_sub = test_sub.drop('fare_binned',axis = 1)
one_hot = pd.get_dummies(test_sub['age_binned'])
test_sub = test_sub.join(one_hot)
test_sub = test_sub.drop('age_binned',axis = 1)


# In[37]:


#def func(row):
#    if row['age_binned'] == '(0.419, 19.0]':
#        return 1
#    elif row['age_binned'] =='(40.5, 80.0]':
#        return 2 
#    elif row['age_binned'] =='(32.5, 40.5]':
#        return 3 
#    elif row['age_binned'] =='(26.0, 32.5]':
#        return 4
#    elif row['age_binned'] =='(25.0, 26.0]':
#        return 5
#    elif row['age_binned'] =='(19.0, 25.0]':
#        return 6
#train_sub['age_bins'] = train_sub.apply(func,axis=1)


# In[38]:


train_sub["age_1"] = train_sub["age_1"].astype('category')
train_sub["age_2"] = train_sub["age_2"].astype('category')
train_sub["age_3"] = train_sub["age_3"].astype('category')
train_sub["age_4"] = train_sub["age_4"].astype('category')
train_sub["age_5"] = train_sub["age_5"].astype('category')
train_sub["age_6"] = train_sub["age_6"].astype('category')
train_sub["sex_target_encoded"] = train_sub["sex_target_encoded"].astype('category')
train_sub["Low_Fare"] = train_sub["Low_Fare"].astype('category')
train_sub["Medium_Fare"] = train_sub["Medium_Fare"].astype('category')
train_sub["High_Fare"] = train_sub["High_Fare"].astype('category')


# In[39]:


test_sub["age_1"] = test_sub["age_1"].astype('category')
test_sub["age_2"] = test_sub["age_2"].astype('category')
test_sub["age_3"] = test_sub["age_3"].astype('category')
test_sub["age_4"] = test_sub["age_4"].astype('category')
test_sub["age_5"] = test_sub["age_5"].astype('category')
test_sub["age_6"] = test_sub["age_6"].astype('category')
test_sub["sex_target_encoded"] = test_sub["sex_target_encoded"].astype('category')
test_sub["Low_Fare"] = test_sub["Low_Fare"].astype('category')
test_sub["Medium_Fare"] = test_sub["Medium_Fare"].astype('category')
test_sub["High_Fare"] = test_sub["High_Fare"].astype('category')


# In[40]:


train_sub.dtypes


# In[41]:


test_sub.dtypes


# In[42]:


train_sub.head()


# In[43]:


#Testing Correlation 


# In[44]:


traincor=train_sub.copy()


# In[45]:


#Don't run it for the first time
del traincor['PC3']
del traincor['Medium_Fare']
del traincor['High_Fare']


# In[46]:


traincor["SibSp"] = traincor["SibSp"].astype('int64')
traincor["Parch"] = traincor["Parch"].astype('int64')
traincor["male"] = traincor["male"].astype('int64')
traincor["S"] = traincor["S"].astype('int64')
traincor["PC1"] = traincor["PC1"].astype('int64')
traincor["PC2"] = traincor["PC2"].astype('int64')
#traincor["PC3"] = traincor["PC3"].astype('int64')
traincor["sex_target_encoded"] = traincor["sex_target_encoded"].astype('int64')
traincor["age_1"] = traincor["age_1"].astype('int64')
traincor["age_2"] = traincor["age_2"].astype('int64')
traincor["age_3"] = traincor["age_3"].astype('int64')
traincor["age_4"] = traincor["age_4"].astype('int64')
traincor["age_5"] = traincor["age_5"].astype('int64')
traincor["age_6"] = traincor["age_6"].astype('int64')
traincor["Low_Fare"] = traincor["Low_Fare"].astype('int64')
#traincor["Medium_Fare"] = traincor["Medium_Fare"].astype('int64')
#traincor["High_Fare"] = traincor["High_Fare"].astype('int64')


# In[47]:


#Return for iteration :
# Drop female,PC1
co_op=traincor.corr(method ='pearson')


# In[48]:


co_op.index.values


# In[49]:


co_op['index']=co_op.index
co_op_mlt=co_op.melt(id_vars =['index'], value_vars =['SibSp', 'Parch', 'male', 'S', 'PC1', 'PC2', 'sex_target_encoded',
       'Low_Fare', 'age_1', 'age_2', 'age_3', 'age_4',
       'age_5', 'age_6'] ,
           var_name ='Variable_column', value_name ='Value_column') 


# In[50]:


co_op_mlt['Abs_value']=abs(co_op_mlt['Value_column'])
co_op_mlt['eliminate'] = np.where((co_op_mlt['Variable_column'] == co_op_mlt['index']) , 1, 0)
co_op_mlt_r=co_op_mlt[co_op_mlt.eliminate==0]
del co_op_mlt_r['eliminate']


# In[51]:


co_op_mlt_r.sort_values(by=['Abs_value'],ascending=False)
#Conclusion : Drop either male or female ; Drop either PC1 or High Fare
# Rest of the combinations look fine


# In[52]:


#Removing correlated columns from train_sub
# train_sub.head()
del train_sub['PC3']
del train_sub['Medium_Fare']
del train_sub['High_Fare']


# In[53]:


#Removing correlated columns from test_sub
# test_sub.head()
del test_sub['PC3']
del test_sub['Medium_Fare']
del test_sub['High_Fare']


# In[54]:


#Converting the values of SibSp :
#train_sub.groupby(['SibSp']).size()
def mod_sibsp(val) :
    if(val==0):
        return 0
    elif(val==1):
        return 1
    else :
        return 2
train_sub['sibsp_mod']=train_sub.apply(lambda x: mod_sibsp(x['SibSp']),axis=1)
test_sub['sibsp_mod']=test_sub.apply(lambda x: mod_sibsp(x['SibSp']),axis=1)
print(train_sub.groupby(['sibsp_mod']).size())
print(test_sub.groupby(['sibsp_mod']).size())


# In[55]:


del train_sub['SibSp']
train_sub["sibsp_mod"] = train_sub["sibsp_mod"].astype('category')


# In[56]:


del test_sub['SibSp']
test_sub["sibsp_mod"] = test_sub["sibsp_mod"].astype('category')


# In[57]:


train_sub.head()


# In[58]:


dup_chk=train_sub.duplicated(subset=None, keep='first')
dup_chk=pd.DataFrame(dup_chk)
dup_chk.columns=["Dup"]
dup_chk.groupby(['Dup']).size()


# In[59]:


train_sub_undisturbed=train_sub[dup_chk['Dup']==False]
train_sub_undisturbed.reset_index(drop=True).to_csv("train_cleaned_python_2.csv",index=False)
train_sub_undisturbed.head()


# In[60]:


train_sub_undisturbed.groupby(['Survived']).size()


# In[61]:


#train_sub_undisturbed
#test_sub


# In[62]:


x = train_sub_undisturbed.iloc[:,1:]
y = train_sub_undisturbed.iloc[:,0:1]
from sklearn.utils import shuffle
x,y = shuffle(x,y)
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)
test_sub_m=test_sub.copy()
del test_sub_m['PassengerId']


# In[63]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression 
import statsmodels.api as sm
classifier = LogisticRegression(solver='liblinear',random_state = 0) 
class_lr=classifier.fit(xtrain, ytrain.values.ravel()) 


# In[64]:


#classifier.coef_
coefficients = pd.concat([pd.DataFrame(xtrain.columns),pd.DataFrame(np.transpose(classifier.coef_))], axis = 1)
print(classifier.intercept_)


# In[65]:


y_pred=classifier.predict(xtest)
y_pred


# In[66]:


test_sub_m.head()


# In[67]:


from sklearn.metrics import r2_score
r2_score(ytest,y_pred)


# In[68]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
cm


# In[69]:


from sklearn.metrics import accuracy_score 
accuracy_score(ytest, y_pred)


# In[70]:


from sklearn.metrics import roc_auc_score
logit_roc_auc = roc_auc_score(ytest, y_pred)


# In[71]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(ytest, y_pred)


# In[72]:


#precision based
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.loc[(roc.tf-0).abs().argsort()[:1]]


# In[73]:


import matplotlib.pyplot as plt 
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])


# In[74]:


cola=test_sub['PassengerId'].tolist()
colb=y_pred.tolist()
#Export Results
export=pd.DataFrame(list(zip(cola,colb)))
export.columns=('PassengerId','Survived')
export.reset_index(drop=True).to_csv("Output_LR.csv",index=False)


# In[75]:


#ROC Curve
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
get_ipython().run_line_magic('matplotlib', 'inline')
y_pred_proba = classifier.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba)
auc = metrics.roc_auc_score(ytest, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[76]:


from sklearn.metrics import classification_report
#print(classification_report(ytest, y_pred))


# In[77]:


# yourValue = randomNumber
# for cols in df.columns:
#     if (yourValue in df[cols]:
#         print('Found in '+cols) #to print the column name if found


# In[78]:


#Decision Tree
#Building Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
#Let's create a Decision Tree Model using Scikit-learn.

# Create Decision Tree classifer object
clf_dt = DecisionTreeClassifier(min_samples_leaf=5,splitter='best')#,criterion='gini')

# Train Decision Tree Classifer
clf_dt.fit(xtrain,ytrain)

#Predict the response for test dataset
#y_pred = clf.predict(xtest)
y_pred = clf_dt.predict(test_sub_m)


# In[79]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Users/antony.morais/Desktop/Amalraj/graphviz-2.38/bin'


# In[80]:


feature_names = [i for i in xtrain.columns]
from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(clf_dt, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)


# In[81]:


cola=test_sub['PassengerId'].tolist()
colb=y_pred.tolist()
#Export Results
export=pd.DataFrame(list(zip(cola,colb)))
export.columns=('PassengerId','Survived')
export.reset_index(drop=True).to_csv("Output_DT.csv",index=False)


# In[82]:


#Feature Importance for Decision Tree
from sklearn.tree import DecisionTreeClassifier
feat_imp=pd.DataFrame(list(zip(xtest.columns,clf_dt.feature_importances_)),columns=['Feature','Importance'])


# In[83]:


feat_imp.sort_values(by=['Importance'],ascending=False)
#We can try removing age_3, sex_target_encoded,Low_Fare, Medium_Fare.age_5


# In[84]:


#cm1=confusion_matrix(ytest, y_pred) 
#cm1
#accuracy_score(ytest, y_pred)


# In[85]:


x = train_sub_undisturbed.iloc[:,1:]
y = train_sub_undisturbed.iloc[:,0:1]
from sklearn.utils import shuffle
x,y = shuffle(x,y)
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)
test_sub_m=test_sub.copy()
del test_sub_m['PassengerId']


# In[86]:


#RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=4,random_state=5)
# fit the model with the training data
model.fit(xtrain,ytrain.values.ravel())
y_pred=model.predict(test_sub_m)
cola=test_sub['PassengerId'].tolist()
colb=y_pred
#Export Results
export=pd.DataFrame(list(zip(cola,colb)))
export.columns=('PassengerId','Survived')
export.reset_index(drop=True).to_csv("Output_Nov19RF.csv",index=False)


# In[87]:


#Gradient Boosting
from sklearn import ensemble
from sklearn import linear_model


# In[88]:


params = {
     'n_estimators': 15,
     'max_depth': 10,
     'learning_rate': 0.14
}
gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)
gradient_boosting_fitted=gradient_boosting_regressor.fit(xtrain,ytrain.values.ravel())
#Predict the response for test dataset
y_pred = gradient_boosting_fitted.predict(xtest)
y_pred_fin = [1 if i >=0.5 else 0 for i in y_pred]


# In[89]:


#cm1=confusion_matrix(ytest, y_pred_fin) 
#print(cm1)
#accuracy_score(ytest, y_pred_fin)


# In[90]:


########################################  Interpretability ########################################
# Example on PD plot in sklearn library
#It can be used only for gradient boosting model

#The assumption of independence is the biggest issue with PD plots. 
#It is assumed that the feature(s) for which the partial dependence is computed are not correlated with other features

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
features = ['Parch','male']  # Features used for computing and plotting PDPlot
fig, axs =  plot_partial_dependence(gradient_boosting_regressor,       
                                   features=[0, 1, 2,(1,2)], # column numbers of plots we want to show
                                   X=x,            # raw predictors data.
                                   feature_names=['Parch', 'male', 'S'],grid_resolution=10) # labels on graphs


# In[91]:


# Import functions - Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
# Get variables for which to compute VIF and add intercept term
X = test_sub_m.copy()


# In[92]:


# Compute and view VIF
X['Intercept'] = 1
vif = pd.DataFrame()
vif["variables"] = X.columns
#The input to Variance Calculation must be an integer
yy = np.array(X.values, dtype='int32')
vif["VIF"] = [variance_inflation_factor(yy,i) for i in range(X.shape[1])]

# View results using print
print(vif)


# In[93]:


# XGBoost
#import xgboost as xgb
from xgboost import XGBClassifier


# In[94]:


x['Parch']=x['Parch'].astype('int64')
x['male']=x['male'].astype('int64')
x['female']=x['female'].astype('int64')
x['S']=x['S'].astype('int64')
x['PC1']=x['PC1'].astype('int64')
x['PC2']=x['PC2'].astype('int64')
x['sex_target_encoded']=x['sex_target_encoded'].astype('int64')
x['age_1']=x['age_1'].astype('int64')
x['age_2']=x['age_2'].astype('int64')
x['age_3']=x['age_3'].astype('int64')
x['age_4']=x['age_4'].astype('int64')
x['age_5']=x['age_5'].astype('int64')
x['age_6']=x['age_6'].astype('int64')
x['sibsp_mod']=x['sibsp_mod'].astype('int64')
x['Low_Fare']=x['Low_Fare'].astype('int64')
y['Survived']=y['Survived'].astype('int64')


# In[95]:


x.dtypes


# In[96]:


#!pip install sklearn.cross_validation
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[97]:


xg_class = XGBClassifier(earning_rate=0.3, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=7,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0)


# In[98]:


xg_class.fit(xtrain,ytrain.values.ravel())
print(xg_class)


# In[99]:


x.dtypes


# In[100]:


test_sub1=test_sub_m.copy()
test_sub1['Parch']=test_sub1['Parch'].astype('int64')
test_sub1['male']=test_sub1['male'].astype('int64')
test_sub1['female']=test_sub1['female'].astype('int64')
test_sub1['S']=test_sub1['S'].astype('int64')
test_sub1['PC1']=test_sub1['PC1'].astype('int64')
test_sub1['PC2']=test_sub1['PC2'].astype('int64')
test_sub1['sex_target_encoded']=test_sub1['sex_target_encoded'].astype('int64')
test_sub1['age_1']=test_sub1['age_1'].astype('int64')
test_sub1['age_2']=test_sub1['age_2'].astype('int64')
test_sub1['age_3']=test_sub1['age_3'].astype('int64')
test_sub1['age_4']=test_sub1['age_4'].astype('int64')
test_sub1['age_5']=test_sub1['age_5'].astype('int64')
test_sub1['age_6']=test_sub1['age_6'].astype('int64')
test_sub1['sibsp_mod']=test_sub1['sibsp_mod'].astype('int64')
test_sub1['Low_Fare']=test_sub1['Low_Fare'].astype('int64')
preds = xg_class.predict(test_sub1)


# In[101]:


y_pred = [1 if i >= 0.5 else 0 for i in preds]
#y_pred


# In[102]:


#accuracy = accuracy_score(ytest, y_pred)
#accuracy


# In[103]:


cola=test_sub['PassengerId'].tolist()
colb=y_pred
#Export Results
export=pd.DataFrame(list(zip(cola,colb)))
export.columns=('PassengerId','Survived')
export.reset_index(drop=True).to_csv("Output_Nov19xgb.csv",index=False)


# In[104]:


################   Jump to SHAP Test and then come back to LightGBM  ##################


# In[105]:


#Light GBM


# In[106]:


import lightgbm as lgb


# In[136]:


xtrain.columns


# In[137]:


xtrain=xtrain[['Parch','male','S','PC2','PC1','age_3','age_1','age_6','age_4','age_2','sibsp_mod']]


# In[138]:


d_train = lgb.Dataset(xtrain, label=ytrain.values.ravel())
params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
#params['num_boost_round'] = 20
lgb_m = lgb.train(params, d_train, 100)


# In[139]:


y_pred=lgb_m.predict(test_sub1)


# In[140]:


y_pred=[1 if i>=0.56 else 0 for i in y_pred]


# In[141]:


#confusion_matrix(ytest, ypred)


# In[142]:


#accuracy_score(ypred,ytest)


# In[143]:


cola=test_sub['PassengerId'].tolist()
colb=y_pred
#Export Results
export=pd.DataFrame(list(zip(cola,colb)))
export.columns=('PassengerId','Survived')
export.reset_index(drop=True).to_csv("Output_Nov19lgb.csv",index=False)


# In[144]:


####################################  LIME  ##############################################
import lime
import lime.lime_tabular


# In[145]:


explainer = lime.lime_tabular.LimeTabularExplainer(xtrain[lgb_m.feature_name()].astype(float).values,  
mode='classification',training_labels=ytrain['Survived'],feature_names=lgb_m.feature_name())


# In[146]:


xtrain.head()


# In[147]:


ytrain.head()


# In[148]:


xtrain.reset_index(drop=True,inplace=True)
ytrain.reset_index(drop=True,inplace=True)


# In[149]:


# asking for explanation for LIME model
i = 185
def prob(data):
    return np.array(list(zip(1-lgb_m.predict(data),lgb_m.predict(data))))


# In[150]:


exp_ip=xtrain.loc[i,feat].astype(float).values


# In[151]:


exp = explainer.explain_instance(exp_ip, prob)


# In[152]:


exp.show_in_notebook(show_table=True)

############################################     SHAP     ######################################
#!pip install shap


# In[158]:


import shap



# In[159]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
# data argument is optional when feature_perturbation=”tree_path_dependent”. But, by default this value will be "interventional"
explainer = shap.TreeExplainer(model=xg_class)#,data=xtest,feature_perturbation="interventional")


# In[160]:


shap_values = explainer.shap_values(xtest)


# In[161]:


### Data point to he explained
i = 75


# In[168]:


xtest.iloc[i,]


# In[169]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[i,:], xtest.iloc[i,:])
#The “base value” (see the grey print towards the upper-left of the image) marks the model’s average prediction over the training set.
# The “output value” is the model’s prediction. The feature values for the largest effects are printed at the bottom of the plot. 

# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, xtest)

# create a dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("Low_Fare", shap_values, xtest)



# summarize the effects of all the features
shap.summary_plot(shap_values,xtest)

# In[175]:

shap.summary_plot(shap_values, xtest, plot_type="bar")