# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 21:17:02 2019

@author: Jahnvi Patel
"""

from random import seed
import numpy as np
from random import randrange
import pandas as pd
import csv
from csv import reader
df = pd.DataFrame()

data = pd.read_csv("train_modified101.csv")
#dt = data.head()
#print(data.head())
#data['Item_Identifier'] = data['Item_Identifier'].astype(str)
data['Item_Identifier'] = data['Item_Identifier'].astype(str)
data['Outlet_Identifier'] = data['Outlet_Identifier'].astype(str)
x = np.array(data['Item_Identifier'])
y = np.array(data['Outlet_Identifier'])
n=0
for a1 in x:
    z = [ord(c) for c in a1]
    z1=[int(i) for i in z]
    z2 = [str(i) for i in z1]
    z3=''.join(z2)
    z4 = int(z3)
    #print(z1)
    data['Item_Identifier'][n]=z4
    n=n+1

n=0
for a2 in y:
    b = [ord(c) for c in a2]
    b1 = [int(i) for i in b]
    b2 = [str(i) for i in b1]
    b3 = ''.join(b2)
    b4 = int(b3)
    data['Outlet_Identifier'][n]=b4
    #data['Outlet_Identifier'][2] = 23456
    n=n+1
#data['Outlet_Identifier'][1]=8989898989
'''with open('test_modified.csv', 'w') as f:
    writer = csv.writer(f)
    for val in data['Item_Identifier']:
        writer.writerow([val])'''
data.to_csv("test_modified.csv")


#[float(i) for i in z]
#print(z)

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt",encoding='utf8')
    lines = reader(file)
    dataset = list(lines)
    return dataset




# Convert string column to float
    '''
def str_column_to_float(dataset, column):
    for row in dataset:
       # print(row[1])
        print(row[column])
'''


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split



def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0



def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']



def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)



seed(1)

dataset = load_csv("test_modified.csv")

n=0
#for i in range(len(dataset[0])):
    #if n!=0:
        #str_column_to_float(dataset, i)
    #n=n+1

n_folds = 6
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
