# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:00:40 2019

@author: Jahnvi Patel
"""

#Importing the libraries required
import pandas as pd
import numpy as np

#Read files:
train = pd.read_csv("Train_final.csv")
#test = pd.read_csv("Test_final.csv")

train['source']='train'
#test['source']='test'
data = pd.concat([train],ignore_index=True)
print (train.shape, data.shape)

new=data.apply(lambda x: sum(x.isnull()))
print(new)
print(data.describe())
new1=data.apply(lambda x: len(x.unique()))
print(new1)

#Filter categorical variables
#categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
#categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
#for col in categorical_columns:
    #print ('\nFrequency of Categories for varible %s'%col)
    #print (data[col].value_counts())

#let do grouping in each catogorical columns

col=["Item_Fat_Content","Item_Type","Outlet_Location_Type","Outlet_Size","Outlet_Type"]

for i in col:
    print("The frequency distribution of each catogorical columns is--" + i+"\n")
    print(data[i].value_counts())

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull()
print ('Orignal missing: %d'% sum(miss_bool))

#Replacing the minimum nan values in the Item_Weight with its mean value
data.fillna({"Item_Weight":data["Item_Weight"].mean()},inplace=True)

#checking the current status of  nan values in the dataframe
data.apply(lambda x: sum(x.isnull()))
print ('Final missing: %d'% sum(data['Item_Weight'].isnull()))

#Get a boolean variable specifying missing Outlet_Size values
miss_bool = data['Outlet_Size'].isnull()
print ('\nOrignal missing: %d'% sum(miss_bool))

#Now we have 0 nan values in Outlet_Size
data["Outlet_Size"].fillna(method="ffill",inplace=True)
data.apply(lambda x: sum(x.isnull()))
print ('Final missing: %d'% sum(data['Outlet_Size'].isnull()))

#Check the mean sales by type:
#print(data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type'))

#Now working on the item_visibility
#Determine average visibility of a product
#Get all Item_Visibility mean values for respective Item_Identifier
visibility_item_avg = data.pivot_table(values='Item_Visibility',index='Item_Identifier')

def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index == item]
    else:
        return visibility

print ('Original #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'] = data[['Item_Visibility','Item_Identifier']].apply(impute_visibility_mean,axis=1).astype(float)
print ('Final zeros: %d'%sum(data['Item_Visibility'] == 0))

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print(data['Item_Type_Combined'].value_counts())

#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#Change categories of low fat:
print('Original Categories:')
print(data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
print(data['Item_Fat_Content'].value_counts())

#Implementing one-hot-Coding method for getting the categorical variables
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
#print(data.dtypes)

data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

train = data.loc[data['source']=="train"]

train.drop(['source'],axis=1,inplace=True)

train.to_csv("train_modified101.csv",index=False)



