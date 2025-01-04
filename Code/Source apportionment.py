import pandas as pd
import numpy as np
import pickle
import os
import glob
import imageio
import time
import random
import sys
import copy
import json
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import gc
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score, roc_curve, auc, classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold, cross_validate
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import warnings


# ## Conduct source apportionment of the particulate matter on the training set.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("../Trained model/ResNet_V2.pth")
model.to(device)
model.eval()

data_dir = '../Row data/Training data/'
train_dir = data_dir + 'train/'
test_dir = data_dir + 'test/'

val_tf = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(train_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

df = pd.DataFrame()
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()
    df = df._append(pd.DataFrame(class_probabilities), ignore_index=True)
image_datasets = datasets.ImageFolder(test_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

df2 = pd.DataFrame()
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()
    df2 = df2._append(pd.DataFrame(class_probabilities), ignore_index=True)

# Merging with EDS and size data
pickle_file = '../Row data/train_test_data_2.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    X_train = pickle_data['train_dataset']
    y_train = pickle_data['train_labels']
    X_test = pickle_data['test_dataset']
    y_test = pickle_data['test_labels']
    del pickle_data
    gc.collect()
print('The data has been loaded')

X_train = np.hstack([np.array(df),X_train])
X_test = np.hstack([np.array(df2),X_test])

le = LabelEncoder()
y_train_le =  le.fit_transform(y_train)
y_test_le =  le.fit_transform(y_test)
le.classes_


f = open("../Trained model/RX_V2.pickle", 'rb')
RX_model = pickle.load(f)
f.close()

y_pred = RX_model.predict(X_test)
y_pred_prob = RX_model.predict_proba(X_test)

pd_test_data = pd.read_excel('../Row data/train_test_row_data.xlsx', sheet_name = "test_data")
pd_select = pd_test_data[['Source', 'Mass', 'Davg']]
probability = y_pred_prob
mass = np.array(pd_select['Mass']).reshape(-1,1)
probability_mass = probability * mass
Source_type = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']
True_num_contribution = pd_select['Source'].value_counts().reindex(Source_type).values / pd_select['Source'].value_counts().reindex(Source_type).values.sum()
True_mass_contribution_ = pd_select.groupby('Source').agg({'Mass': 'sum'})
True_mass_contribution = np.array(True_mass_contribution_['Mass']/True_mass_contribution_['Mass'].sum())
print(True_num_contribution)
print(True_mass_contribution)


num_contribution = probability.sum(axis = 0) / probability.sum()
mass_contribution = probability_mass.sum(axis = 0) / probability_mass.sum()
print(num_contribution)
print(mass_contribution)

def relative_error(A, B):
    return(np.abs(A - B)/ B)
num_con_error = relative_error(num_contribution, True_num_contribution)
mass_con_error = relative_error(mass_contribution, True_mass_contribution)
print(num_con_error)
print(mass_con_error)


bins = [0.2, 1.0, 2.5, 5.0, 7.5, 10]
labels = ['0.2-1.0', '1.0-2.5', '2.5-5', '5-7.5', '7.5-10']


pd_select['bin'] = pd.cut(pd_select['Davg'], bins=bins, labels=labels, right=False)
pd_conbined_num = pd.concat([pd_select, pd.DataFrame(probability, columns = Source_type)], axis = 1)
pd_conbined_mass = pd.concat([pd_select, pd.DataFrame(probability_mass, columns = Source_type)], axis = 1)

bin_num_pre = pd_conbined_num.iloc[:,3: ].groupby('bin').sum()
bin_mass_pre = pd_conbined_mass.iloc[:,3: ].groupby('bin').sum()
bin_num_pre = bin_num_pre.div(bin_num_pre.sum(axis=1), axis=0)
bin_mass_pre = bin_mass_pre.div(bin_mass_pre.sum(axis=1), axis=0)

bin_num_true = (pd_conbined_num.groupby('bin')['Source']
                .value_counts()
                .rename('count')
                .reset_index()
                .assign(total_count=lambda x: x.groupby('bin')['count'].transform('sum'),
                        perc=lambda x: x['count'] / x['total_count']))
bin_num_true = bin_num_true.pivot_table(index='bin', columns='Source', values='perc').reindex(columns = Source_type)
print(bin_num_true)

bin_mass_true = (pd_conbined_mass.groupby(['bin', 'Source'])
                 .agg({'Mass': 'sum'})
                 .reset_index()
                 .assign(total_mass=lambda x: x.groupby('bin')['Mass'].transform('sum'),
                         perc=lambda x: x['Mass'] / x['total_mass']))
bin_mass_true = bin_mass_true.pivot_table(index='bin', columns='Source', values='perc').reindex(columns = Source_type)
print(bin_mass_true)

bin_num_con_error = relative_error(np.array(bin_num_true), np.array(bin_num_pre))
bin_mass_con_error = relative_error(np.array(bin_mass_true), np.array(bin_mass_pre))

print(pd.DataFrame(bin_num_con_error, index = labels, columns = Source_type))

num_con_error_df = pd.DataFrame([num_con_error.tolist()], columns=Source_type, index=['Mean_num_error'])
mass_con_error_df = pd.DataFrame([mass_con_error.tolist()], columns=Source_type, index=['Mean_mass_error'])
Error_num = pd.concat([pd.DataFrame(bin_num_con_error, index = labels, columns = Source_type), num_con_error_df], axis = 0)
Error_mass = pd.concat([pd.DataFrame(bin_mass_con_error, index = labels, columns = Source_type), mass_con_error_df], axis = 0)
print(Error_num)
print(Error_mass)

writer = pd.ExcelWriter("../Out put/Source apportionment errors.xlsx")
Error_num.to_excel(writer, sheet_name = 'Error_num', index = True)
Error_mass.to_excel(writer, sheet_name = 'Error_mass', index = True)
writer._save()