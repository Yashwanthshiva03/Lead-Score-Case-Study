#!/usr/bin/env python
# coding: utf-8

# #                                     Lead Scoring Case Study_ML_1

# ### An education company named X Education sells online courses to industry professionals - Finding Lead Score on this X Education company...

# In[ ]:





#  Importing the Required Libraries

# In[136]:


# Importing the Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[137]:


# Importing warnigs

import warnings
warnings.filterwarnings('ignore')


# # Reading the Data

# In[138]:


# Reading the csv file

df_leads = pd.read_csv('Leads.csv') 
df_leads.head()


# In[139]:


# Check shape of the DF
df_leads.shape


# In[140]:


# Check statistical data of numerical columns in DF
df_leads.describe()


# # Data Cleaning

# In[141]:


df_leads.info()


# In[142]:


# Check the different levels of all columns
# Get the value counts of all the columns

for column in df_leads:
    print(df_leads[column].value_counts())
    print('-------------------------------------------')


# - Going through the data, we can see there are a few columns in which there is a level called 'Select' which basically means that the student had not selected the option for that particular column which is why it shows 'Select'. 
# - These values are as good as missing values and hence we can replace these with NaN.

# In[143]:


# replace 'Select' with NaN
df_leads.replace('Select',np.NaN, inplace=True)


# In[144]:


# Checkk the count of null values
df_leads.isnull().sum()


# In[145]:


# Check the percentage of missing values in each column

round((df_leads.isnull().sum()/len(df_leads))*100,2)


# In[146]:


# Drop the columns with missing value percentage more than 40%
null_col = ['Asymmetrique Profile Score','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Activity Index','Lead Profile','Lead Quality','How did you hear about X Education']
df_leads.drop(null_col,axis=1,inplace=True)


# In[147]:


# Check the percentage of missing values after dropping columns with high missing values
round((df_leads.isnull().sum()/len(df_leads))*100,2)


# - Also notice that when you got the value counts of all the columns, there were a few columns in which only one value was majorly present for all the data points. 
# - These include Do Not Email, Do Not Call, Search, Magazine, Newspaper Article, X Education Forums, Newspaper, Digital Advertisement, Through Recommendations, Receive More Updates About Our Courses, Update me on Supply Chain Content, Get updates on DM Content, I agree to pay the amount through cheque. 
# - Since practically all of the values for these variables are No, it's best that we drop these columns as they won't help with our analysis.

# In[148]:


# Dropping these highly skewed columns as the won't be useful for analysis

df_leads.drop(['Do Not Email','Do Not Call', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement', 'Through Recommendations', 'Receive More Updates About Our Courses', 
            'Update me on Supply Chain Content', 'Get updates on DM Content', 
            'I agree to pay the amount through cheque'],axis=1,inplace=True)


# Also, the variable 'What matters most to you in choosing a course' has the level 'Better Career Prospects' 6528 times while the other two levels appear once twice and once respectively. So we should drop this column as well.

# In[149]:


df_leads.drop('What matters most to you in choosing a course',axis=1,inplace=True)


# In[150]:


# Checking the null value percentage after dropping unusefull columns
round((df_leads.isnull().sum()/len(df_leads))*100,2)


# In[151]:


# Imputing the missing values in Lead Source with suitable statistical method
df_leads['Lead Source'].fillna(df_leads['Lead Source'].mode()[0],inplace=True)
df_leads['Lead Source'].isnull().sum()


# In[152]:


# Imputing the missing values in Lead Source with suitable statistical method
df_leads['TotalVisits'].fillna(round(df_leads['TotalVisits'].median(),2),inplace=True)
df_leads['TotalVisits'].isnull().sum()


# There is a huge value of null variables in 4 columns as seen above. But removing the rows with the null value will cost us a lot of data and they are important columns. So, instead we are going to replace the NaN values with 'not provided'. This way we have all the data and almost no null values. In case these come up in the model, it will be of no use and we can drop it off then.

# In[153]:


df_leads['Country'].fillna('Not provided',inplace=True)
df_leads['Specialization'].fillna('Not provided',inplace=True)
df_leads['What is your current occupation'].fillna('Not provided',inplace=True)
df_leads['City'].fillna('Not provided',inplace=True)
df_leads['Tags'].fillna('Not provided',inplace=True)


# In[154]:


# Checking the null value percentage after dropping unusefull columns
round((df_leads.isnull().sum()/len(df_leads))*100,2)


# In[155]:


# Dropping the rows with missing values in Page Views Per Visit
df_leads = df_leads[~pd.isnull(df_leads['Page Views Per Visit'])]


# In[156]:


# Checking the null value percentage after dropping unusefull columns
round((df_leads.isnull().sum()/len(df_leads))*100,2)


# In[157]:


df_leads.shape


# In[158]:


df_leads['City'].value_counts()


# In[159]:


# Checking the number of unique categorical levels

for column in df_leads:
    print(df_leads[column].value_counts())
    print('-------------------------------------------')


# In some columns there are couple categorie/Levels that contribute a very low percentage of total values. So clubbing them together for better analysis
# 

# In[160]:


# Clubbing the levels with low data percentage in Country

def slots(x):
    category = ""
    if x == "India":
        category = "India"
    elif x == "Not provided":
        category = "Not provided"
    else:
        category = "Others"
    return category


df_leads['Country'] = df_leads.apply(lambda x:slots(x['Country']), axis = 1)
df_leads['Country'].value_counts()


# In[161]:


# # Clubbing the levels with low data percentage in Lead Score

def slots(x):
    category = ""
    if x == "Google":
        category = "Google"
    elif x == "Direct Traffic":
        category = "Direct Traffic"
    elif x == "Olark Chat":
        category = "Olark Chat"
    elif x == "Organic Search":
        category = "Organic Search"    
    elif x == "Reference":
        category = "Reference" 
    elif x == "google":
        category = "Google"
    else:
        category = "Not Provided"
    return category


df_leads['Lead Source'] = df_leads.apply(lambda x:slots(x['Lead Source']), axis = 1)
df_leads['Lead Source'].value_counts()


# In[162]:


# Clubbing the levels with low data percentage in City

def slots(x):
    category = ""
    if x in ('Mumbai','Other Metro Cities'):
        category = "Tier I Cities"
    elif x in ('Thane & Outskirts','Tier II Cities'):
        category = "Tier II Cities"
    elif x in ('Other Cities','Other Cities of Maharashtra'):
        category = "Tier III Cities"
    else:
        category = "Not Provided"
    return category


df_leads['City'] = df_leads.apply(lambda x:slots(x['City']), axis = 1)
df_leads['City'].value_counts()


# In[163]:


# Clubbing the levels with low data percentage in What is your current occupation

def slots(x):
    category = ""
    if x == "Unemployed":
        category = "Unemployed"
    elif x == "Working Professional":
        category = "Working Professional"
    elif x == "Student":
        category = "Student"
    elif x == "Not provided":
        category = "Not provided"
    else:
        category = "Others"
    return category

df_leads['What is your current occupation'] = df_leads.apply(lambda x:slots(x['What is your current occupation']),axis=1)
df_leads['What is your current occupation'].value_counts()


# In[164]:


# Clubbing the levels with low data percentage in Specialization

def slots(x):
    category = ""
    if x == "Finance Management":
        category = "Finance Management"
    elif x == "Human Resource Management":
        category = "Human Resource Management"
    elif x == "Marketing Management":
        category = "Marketing Management"
    elif x == "Operations Management":
        category = "Operations Management"
    elif x == "Not provided":
        category = "Not provided"
    else:
        category = "Others"
    return category

df_leads['Specialization'] = df_leads.apply(lambda x:slots(x['Specialization']),axis=1)
df_leads['Specialization'].value_counts()


# In[165]:


# Dropping columns that do not add any value to analysis
cols = ['Tags','Prospect ID','Lead Number']
df_leads.drop(cols,axis=1,inplace=True)

df_leads.info()


# In[166]:


# Checking the number of rows retained after data cleaning

len(df_leads)/9240*100


# We have retained ~98.5% data 

# In[167]:


# Checking the data for any outliers

df_leads.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])


# It looks like we have some outliers present. Lets check these columns using box plots.

# In[168]:


# Visualizing TotalVisits, Total Time Spent on Website, Page Views Per Visit using boxplot 

plt.figure(figsize=[8,10])
plt.subplot(3,1,1)
sns.boxplot(df_leads['TotalVisits'])

plt.subplot(3,1,2)
sns.boxplot(df_leads['Total Time Spent on Website'])

plt.subplot(3,1,3)
sns.boxplot(df_leads['Page Views Per Visit'])
plt.show()


# Assuming that a customer visiting the page more than 30 times is a rare scenario, lets drop the values >30 in TotalVisits

# In[169]:


df_leads = df_leads[~(df_leads['TotalVisits']>30)]


# In[170]:


# Again checking for outliers

plt.figure(figsize=[8,10])
plt.subplot(3,1,1)
sns.boxplot(df_leads['TotalVisits'])

plt.subplot(3,1,2)
sns.boxplot(df_leads['Total Time Spent on Website'])

plt.subplot(3,1,3)
sns.boxplot(df_leads['Page Views Per Visit'])
plt.show()


# In[171]:


# Univariate Analysis of Categorical variables
cat = ['Lead Origin','Lead Source','Country','A free copy of Mastering The Interview','City','What is your current occupation','Specialization']
x = 1
plt.figure(figsize=(13,40))
for i in cat:
    plt.subplot(4,2,x)
    sns.countplot(df_leads[i])
    plt.xticks(rotation=45)
    plt.title(i)
    x = x+1


# In[172]:


# Visualizing Last Activity & Last Notable Activity wrt Converted
plt.figure(figsize=(15,15))

plt.subplot(2,1,1)
sns.countplot(df_leads['Last Activity'],hue=df_leads['Converted'])
plt.xticks(rotation=45)
plt.title('Last Activity')

plt.subplot(2,1,2)
sns.countplot(df_leads['Last Notable Activity'],hue=df_leads['Converted'])
plt.xticks(rotation=45)
plt.show()


# In[173]:


# Visualizing Numerical variables wrt Converted

plt.figure(figsize=(20,20))
plt.subplot(4,3,1)
sns.barplot(y = 'TotalVisits', x='Converted', palette='Set2', data = df_leads)
plt.subplot(4,3,2)
sns.barplot(y = 'Total Time Spent on Website', x='Converted', palette='Set2', data = df_leads)
plt.subplot(4,3,3)
sns.barplot(y = 'Page Views Per Visit', x='Converted', palette='Set2', data = df_leads)
plt.show()


# In[174]:


# Multivariate Analysis
# Visualizing Lead source wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='Lead Source',hue='Converted',data=df_leads)
plt.show()


# In[175]:


# Visualizing Lead origi wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='Lead Origin',hue='Converted',data=df_leads)
plt.show()


# In[176]:


# Visualizing Country wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='Country',hue='Converted',data=df_leads)
plt.show()


# In[177]:


# Visualizing City wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='City',hue='Converted',data=df_leads)
plt.show()


# In[178]:


# Visualizing What is your current occupation wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='What is your current occupation',hue='Converted',data=df_leads)
plt.show()


# In[179]:


# Visualizing Specialization wrt converted

plt.figure(figsize=(8,5))
sns.countplot(x='Specialization',hue='Converted',data=df_leads)
plt.xticks(rotation=45)
plt.show()


# In[180]:


# Checking for correlation between Numerical variables

Var = df_leads[['TotalVisits','Page Views Per Visit','Total Time Spent on Website','Converted']]

plt.figure(figsize=(10,10))
sns.pairplot(Var,hue='Converted',diag_kind='kde')
plt.show()


# # Data Preparation

# #### Creating dummies for categorical variables

# In[181]:


# Creating dummies for all Categorical variables

cols = ['Lead Origin', 'Lead Source', 'Last Activity','Country', 'Specialization', 'What is your current occupation', 'City','A free copy of Mastering The Interview', 'Last Notable Activity']
new_df = pd.get_dummies(df_leads[cols],prefix=cols,drop_first=True)


# In[182]:


new_df.info()


# In[183]:


df_leads = pd.concat([df_leads,new_df],axis=1)
df_leads.drop(cols,axis=1,inplace=True)


# In[184]:


df_leads.info()


# #### Splitting the Train - Test data

# In[185]:


# Splitting the data into Train - Test data sets

from sklearn.model_selection import train_test_split

df_leads_train, df_leads_test = train_test_split(df_leads, train_size=0.7,test_size=0.3, random_state=100)


# In[186]:


df_leads_train.shape


# In[187]:


df_leads_test.shape


# #### Scaling of Train data

# In[188]:


from sklearn.preprocessing import MinMaxScaler

num_col = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']

scaler = MinMaxScaler()
df_leads_train[num_col] = scaler.fit_transform(df_leads_train[num_col])
df_leads_train.head()


# In[189]:


df_leads_train.describe()


# All the values in numerical columns have been scaled between 0 to 1

# # Data Modeling

# #### Creating X and y data sets

# In[190]:


y_train = df_leads_train.pop('Converted')
X_train = df_leads_train


# In[191]:


X_train.head()


# In[192]:


y_train.head()


# In[193]:


# Checking the variables for correlation
plt.figure(figsize=(20,20))
sns.heatmap(X_train.corr(),cmap='Blues')
plt.show()


# As we can see there a couple columns highly correlated. Droping such columns to prevent multicollinearity.

# In[194]:


drop_cols = ['Last Notable Activity_Email Received','Last Notable Activity_Email Marked Spam','Last Notable Activity_View in browser link Clicked','Last Notable Activity_Resubscribed to emails','Last Notable Activity_Email Opened','Last Notable Activity_Form Submitted on Website','Country_Not provided','Specialization_Not provided','Last Notable Activity_View in browser link Clicked']

X_train.drop(drop_cols,axis=1,inplace=True)


# In[195]:


# Checking the variables for correlation
plt.figure(figsize=(20,20))
sns.heatmap(X_train.corr(),cmap='Blues')
plt.show()


# #### We will be using RFE and statsmodel method combined for building the final model

# In[196]:


# importing necesary libraries

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature seelction using RFE 

LogReg = LogisticRegression()
rfe = RFE(LogReg,15)
rfe = rfe.fit(X_train,y_train)


# In[197]:


list(zip(X_train,rfe.support_,rfe.ranking_))


# In[198]:


column = X_train.columns[rfe.support_]


# In[199]:


column


# In[200]:


# Using Statsmodels detailed statistics
# importing necessary statsmodels library
import statsmodels.api as sm


# #### Model 1

# In[201]:


X_train_rfe = X_train[column]


# In[202]:


X_train_sm = sm.add_constant(X_train_rfe)
lrm1 = sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res1 = lrm1.fit()
res1.summary()


# In[203]:


# Importing Varaiance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Checking VIF values

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Model 2

# In[204]:


# Removing the varaible 'Page Views Per Visit' because of high VIF value ----VIF value <4 considered good enough
X_train_rfe.drop(['Page Views Per Visit'],axis=1,inplace=True)


# In[205]:


# Re-building the model after dropping 'Page Views Per Visit'
X_train_sm = sm.add_constant(X_train_rfe)
lrm2 = sm.GLM(y_train,X_train_sm,family=sm.families.Binomial())
res2 = lrm2.fit()
res2.summary()


# In[206]:


# Checking VIF values

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[207]:


column = X_train_rfe.columns


# All the p-values & VIF values look good. lets assess the model with these variables

# In[208]:


# Assesing the Model

y_train_pred = res2.predict(X_train_sm)
y_train_pred[:10]


# In[209]:


#Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[210]:


y_train_pred_final = pd.DataFrame({'Hot_Lead':y_train.values, 'Hot_Lead_Prob': y_train_pred})
y_train_pred_final.head()


# In[211]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted_0.5'] = y_train_pred_final.Hot_Lead_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[212]:


# Model Evaluation

# importing necessary librarie
from sklearn import metrics


# In[213]:


# Creting confussion matrix

confusion = metrics.confusion_matrix(y_train_pred_final.Hot_Lead, y_train_pred_final['Predicted_0.5'])
confusion


# In[214]:


# Predicted     not_churn    churn
# Actual
# not_churn        3463       455
# churn             732      1715


# In[215]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Hot_Lead, y_train_pred_final['Predicted_0.5'])


# The accuracy is ~81% which is a good value.

# In[216]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[217]:


# Calculating the sensitivity
TP/(TP+FN)


# In[218]:


# Calculating the specificity
TN/(TN+FP)


# ### Optimise Cut off (ROC)

# In[219]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[220]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Hot_Lead, y_train_pred_final.Hot_Lead_Prob, drop_intermediate = False )


# In[221]:


# Call the ROC function
draw_roc(y_train_pred_final.Hot_Lead, y_train_pred_final.Hot_Lead_Prob)


# The area under ROC curve is 0.89, which is a good value

# In[222]:


# Creating columns with different probability cutoffs

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Hot_Lead_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[223]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity

from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Hot_Lead, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[224]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# From the graph we can see that the optimal cutoff is around 0.35

# In[225]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Hot_Lead_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()


# #### Creating lead Score

# Creating a column called Lead Score by multplying the converted probablity with 100.

# In[226]:


# Creating Lead Score column
y_train_pred_final['Lead Score'] = round((y_train_pred_final['Hot_Lead_Prob']*100))
y_train_pred_final.head()


# - The customers with Lead Score more than 35 will be converted as we decided to have the optimal probablity cutoff is 0.35
# - Higher the Lead Score, higher the chance of the customers to be converted.

# ###### Finding the average Lead Score of the predicted converted leads

# In[227]:


# Creating dataframe for predicted converted leads
y_train_pred_converted = y_train_pred_final[y_train_pred_final['final_predicted']==1]
y_train_pred_converted.head()


# In[228]:


# Average Lead Score of the predicted converted leads
avg_converted = round(sum(y_train_pred_converted['Lead Score'])/len(y_train_pred_converted.index))
avg_converted


# We can see that the average Lead Score of the customers, who were converted is 71.

# ##### Finding the average Lead Score of the predicted not converted leads

# In[229]:


# Creating dataframe for predicted not converted leads
y_train_pred_not_converted = y_train_pred_final[y_train_pred_final['final_predicted']==0]
y_train_pred_not_converted.head()


# In[230]:


# Average Lead Score of the predicted not converted leads
avg_not_converted = round(sum(y_train_pred_not_converted['Lead Score'])/len(y_train_pred_not_converted.index))
avg_not_converted


# We can see that the average Lead Score of the customers, who were not converted is 13

# In[231]:


# Bar plot
plt.bar(['Converted', 'Not Converted'], [avg_converted, avg_not_converted])
plt.xlabel('Leads')
plt.ylabel('Lead Score')
plt.title('Avg. Lead Score')
plt.show()


# In[232]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Hot_Lead, y_train_pred_final.final_predicted)


# In[233]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Hot_Lead, y_train_pred_final.final_predicted )
confusion2


# In[234]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[235]:


# Calculating the sensitivity
TP/(TP+FN)


# In[236]:


# Calculating the specificity
TN/(TN+FP)


# ##### With cutoff of 0.35 the Accuracy is ~80%, Sensitivity is ~82% & Specificity is ~80% for train data set

# # Prediction on Test data set

# In[237]:


# Scaling Numeric variables

df_leads_test[num_col] = scaler.transform(df_leads_test[num_col])


# In[238]:


df_leads_test.describe()


# In[239]:


y_test = df_leads_test.pop('Converted')


# In[240]:


# Select the columns in X_train for X_test as well
X_test = df_leads_test[column]

# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)
X_test_sm


# In[241]:


X_test.head()


# In[242]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res2.predict(X_test_sm)

# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Hot_Lead_Prob'})
y_pred_final.head()


# In[243]:


# Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Hot_Lead_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final


# In[244]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[245]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[246]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[247]:


# Calculating the sensitivity
TP/(TP+FN)


# In[248]:


# Calculating the specificity
TN/(TN+FP)


# ##### With cutoff 0.35 Accuracy is ~80%, Sensitivity is ~81% and Specificity is ~80% for test data set, which are pretty close to train data

# ### Precission & Recall tradeoff on train data

# In[249]:


from sklearn.metrics import precision_recall_curve


# In[250]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Hot_Lead, y_train_pred_final.Hot_Lead_Prob)


# In[251]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# Cutoff is around 0.41

# In[252]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Hot_Lead_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()


# In[253]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Hot_Lead, y_train_pred_final.final_predicted)


# In[254]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Hot_Lead, y_train_pred_final.final_predicted )
confusion2


# In[255]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[256]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[257]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ##### With the current cut off as 0.41 we have Precision of ~75% and Recall of ~77% for train data set

# ### Prediction on Test data set

# In[258]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res2.predict(X_test_sm)

# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Hot_Lead_Prob'})
y_pred_final.head()


# In[259]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Hot_Lead_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_pred_final


# In[260]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[261]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[262]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[263]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[264]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ##### With cutoff of 0.41 we have Precission of ~74% and Recall 0f ~76% for the test data set, which are pretty close to the values of train data.

# ### Conclusion
# 
# It was found that the variables that mattered the most in the potential buyers are:
# 
# 1. Total number of visits.
# 2. The total time spend on the Website
# 4. When the Last Notable Activity was:
#     - SMS_Sent
#     - Had a Phone Conversation
#     - Unreachable
# 5. When the lead source was:
#     - Olark Chat
# 6. When lead origin was:
#     - Lead Add Form
# 7. When the last activity was:
#     - Olark chat conversation
#     - Converted to Lead
#     - Email Bounced
# 8. What is their current occupation is as a Unemployed, Working professional, Student & other.
# 
# Keeping these in mind the X Education can flourish as they have a very high chance to get almost all the potential buyers to change their mind and buy their courses.

# In[ ]:




