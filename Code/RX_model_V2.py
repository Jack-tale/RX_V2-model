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
import shutil
from PIL import Image
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import gc
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score, roc_curve, auc, classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold, cross_validate
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import warnings

# ## RX model training

## Calculating the probability values of single particle sources using the trained ResNet model
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


# XGBoost model training

le = LabelEncoder()
y_train_le =  le.fit_transform(y_train)
y_test_le =  le.fit_transform(y_test)
le.classes_

# Simple test
XGB = XGBClassifier()
XGB.fit(X_train, y_train_le)
y_pred = XGB.predict(X_test)
print(accuracy_score(y_test_le, y_pred))


# Parameter_tuning_1

param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "eta": [0.05, 0.1, 0,2, 0.3],
        "max_depth": [3,4,5,6,7,8],
        "colsample_bytree": [0.4,0.6,0.8,1],
        "min_child_weight": [1,2,4,6,8]
     }
XGB = XGBClassifier()
cv = StratifiedKFold(n_splits = 9, shuffle = True)
RX_model = RandomizedSearchCV(XGB, param_distributions = param_grid, cv = cv, n_iter = 50, n_jobs = -1)
starttime = datetime.datetime.now()
RX_model.fit(X_train, y_train_le)
endtime = datetime.datetime.now()

print('Run time:', endtime - starttime)
print(list(zip(RX_model.cv_results_['params'],RX_model.cv_results_['mean_test_score'])))
print("Best params：:", RX_model.best_params_)
print("Best score:", RX_model.best_score_)


# Parameter_tuning_2

param_grid = {
        "n_estimators": [100, 200, 300],
        "eta": [0.03, 0.05, 0.1, 0,2],
        "max_depth": [1, 3, 5],
        "colsample_bytree": [0.4,0.6, 0.8],
        "min_child_weight": [1,3,5]
     }
XGB = XGBClassifier()
cv = StratifiedKFold(n_splits = 9, shuffle = True)
RX_model = GridSearchCV(XGB, param_grid = param_grid, cv = cv, n_jobs = -1)
starttime = datetime.datetime.now()
RX_model.fit(X_train, y_train_le)
endtime = datetime.datetime.now()
print('Run time:', endtime - starttime)
print(list(zip(RX_model.cv_results_['params'],RX_model.cv_results_['mean_test_score'])))
print("Best params：:", RX_model.best_params_)
print("Best score:", RX_model.best_score_)



# The performance of the model on (train, validation) sets

RX_model = RX_model.best_estimator_


scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    'f1_weighted': 'f1_weighted',
    'roc_auc_weighted': 'roc_auc_ovr_weighted',
    'aupr_weighted': make_scorer(average_precision_score, average='weighted', needs_proba=True)
}
cv = StratifiedKFold(n_splits = 9, shuffle = True)
scores = cross_validate(RX_model, X_train, y_train_le, scoring=scoring, cv=cv, return_train_score=False)

# The performance of the model on test sets

y_pred = RX_model.predict(X_test)
y_pred_prob = RX_model.predict_proba(X_test)

accuracy = accuracy_score(y_test_le, y_pred)
precision_weighted = precision_score(y_test_le, y_pred, average='weighted')
recall_weighted = recall_score(y_test_le, y_pred, average='weighted')
f1_weighted = f1_score(y_test_le, y_pred, average='weighted')
roc_auc_weighted = roc_auc_score(y_test_le, y_pred_prob, average='weighted', multi_class='ovr')
aupr_weighted = average_precision_score(y_test_le, y_pred_prob, average='weighted')


metrics = {
    "Metric": ["Accuracy", "Precision_weighted", "Recall_weighted", "F1-Score_weighted", "ROC AUC_weighted", "AUPR_weighted"],
    "Train_dataset(Mean)": [
        scores['test_accuracy'].mean(),
        scores['test_precision_weighted'].mean(),
        scores['test_recall_weighted'].mean(),
        scores['test_f1_weighted'].mean(),
        scores['test_roc_auc_weighted'].mean(),
        scores['test_aupr_weighted'].mean()
    ],
    "Train_dataset(Standard Deviation)": [
        scores['test_accuracy'].std(),
        scores['test_precision_weighted'].std(),
        scores['test_recall_weighted'].std(),
        scores['test_f1_weighted'].std(),
        scores['test_roc_auc_weighted'].std(),
        scores['test_aupr_weighted'].std()
    ],
    "Test_dataset": [
        accuracy,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        roc_auc_weighted,
        aupr_weighted
    ]
}

df_RX = pd.DataFrame(metrics)
print(df_RX)
df_RX.to_excel('../Out put/RX_results.xlsx', index=False, engine='openpyxl')

# save the model
f = open("../Trained model/RX_V2.pickle", 'wb')
pickle.dump(RX_model,f, protocol = pickle.HIGHEST_PROTOCOL)
f.close()

def multi_class_ROC(y_test, predict_proba, model_classes,fig_path = False):
    y_test_bin = label_binarize(y_test, classes=model_classes)
    y_test_pred_proba = predict_proba
    # model_classes = list(model_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    j = 0
    for i in model_classes:
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, j], y_test_pred_proba[:, j])
        roc_auc[i] = auc(fpr[i], tpr[i])
        j += 1

    # Average result
    fpr["average"], tpr["average"], _ = roc_curve(y_test_bin.ravel(), y_test_pred_proba.ravel())
    roc_auc["average"] = auc(fpr["average"], tpr["average"])

    # Plot
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='red', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'cyan', 'greenyellow', 'dodgerblue']
    for i, color in zip(model_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 30)
    plt.ylabel('True Positive Rate', fontsize = 30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.title('multi-calss ROC')
    plt.legend(loc="lower right", fontsize = 18)
    if fig_path:
        plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.show()

multi_class_ROC(y_test,y_pred_prob,le.classes_, fig_path = '../Out put/RX_ROC_curve.pdf')

test_label = le.inverse_transform(y_pred)
predict_results = np.hstack((X_test, y_test.reshape((-1, 1)), test_label.reshape((-1, 1))))

writer = pd.ExcelWriter('../Out put/RX_test_label.xlsx')
pd.DataFrame(predict_results).to_excel(writer, sheet_name = 'test_label', index = False)
writer._save()

data = pd.DataFrame(predict_results).iloc[:, -2:]
labels = data.rename(columns = {34:'True_labels', 35:'Predicted_labels'})
print(labels)

Class = pd.read_excel('../Row data/train_test_row_data.xlsx', sheet_name = 'test_data')[['Class_2']]
Dataframe = Class.join(labels).rename(columns = {'Class_2':'Class'})
print(Dataframe)

mapping = {'Biomass':'Biomass burning', 'Coal':'Coal combustion', 'Construction':'Construction dust', 'Road dust':'Road dust', 'Soil':'Soil', 'Steel':'Steelmaking'}
Dataframe['True_labels'] = Dataframe['True_labels'].replace(mapping)
Dataframe['Predicted_labels'] = Dataframe['Predicted_labels'].replace(mapping)
print(Dataframe)

def con_matrix_normied_plot(data, colorbar_ticks = [0.2, 0.4, 0.6, 0.8], fig_path = False):
    
    labels = ['Soil', 'Road dust', 'Construction dust', 'Coal combustion', 'Biomass burning', 'Steelmaking']
    con_matrix = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels)
    con_matrix_normalized = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels, normalize='true')
                      
    plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(len(labels)), labels=labels, rotation=30, rotation_mode="anchor", ha="right", fontsize = 25)
    plt.xlabel('Predicted source', fontsize = 30)
    plt.yticks(np.arange(len(labels)), labels=labels, fontsize = 25)
    plt.ylabel('True source', fontsize = 30)
    # plt.title("Harvest of local farmers (in tons/year)")
    f1 = lambda x: '%.2f%%' % (x * 100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                text = plt.text(j, i, f1(con_matrix_normalized[i, j]) + "\n" + "(" + str(con_matrix[i, j]) + ")", ha="center", va="center", color="b", fontsize = 22)
            else:
                text = plt.text(j, i, f1(con_matrix_normalized[i, j]) + "\n" + "(" + str(con_matrix[i, j]) + ")", ha="center", va="center", color="w", fontsize = 22)


    
    plt.imshow(con_matrix_normalized)
    
    cax = plt.axes([0.95, 0.12, 0.04, 0.76])
    colorbar = plt.colorbar(cax= cax)
    colorbar.ax.set_yticks(colorbar_ticks)
    colorbar.ax.set_yticklabels(['{:.0%}'.format(x) for x in colorbar_ticks], fontsize = 22)
    #plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.show()
    
con_matrix_normied_plot(Dataframe, fig_path='../Out put/RX_confusion_matrix.png')

mapping = {'Biomass burning':'BB', 'Coal combustion':'CC', 'Construction dust':'CD', 'Road dust':'RD', 'Soil':'So', 'Steelmaking':'St'}
Dataframe['True_labels'] = Dataframe['True_labels'].replace(mapping)
Dataframe['Predicted_labels'] = Dataframe['Predicted_labels'].replace(mapping)
print(Dataframe)

def con_matrix_norm_sub_plot(Class_True_Predicted_labels, rows = 7, cols = 5, colorbar_ticks = [0.2, 0.4, 0.6, 0.8], fig_path = False):
    from matplotlib.colors import ListedColormap
    # import matplotlib
    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(rows, cols, figsize=(22, 30))
    #Class_type = Class_True_Predicted_labels['Class'].unique()
    Class_type = ["Si-Al", "Si-rich","Si-Al-Ca","Si-Al-Fe","Si-Al-K-Mg","Si-Al-Mg","Si-Al-Na","Si-Al-Ca-Mg-Fe","Si-Al-Ca-Mg","Si-Ca","Al-rich","Al-Ca","Ca-Mg","Ca-Mg-Cl","P-containing","K-containing","Fly ash","C-rich","Fe-rich","Ca-rich","S-Ca","Ti-rich","Fe-Ca-S","Fe-K-Cl", "Mn-rich","K-Cl","Pb-Cl","Pb-Fe-Cl","Pb-Cl-K","S-Al-Ca-Mg","S-Ca-Pb","S-Ca-K-Cl","Other"]
    
    im = None
    
    for Class_id in range(len(Class_type)):
                
        data = Class_True_Predicted_labels[Class_True_Predicted_labels['Class'] == Class_type[Class_id]]
        labels = list(set(list(data['True_labels'].unique()) + list(data['Predicted_labels'].unique()) ))
        
        con_matrix = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels)
        con_matrix_normalized = confusion_matrix(data['True_labels'], data['Predicted_labels'], labels = labels, normalize='true')
        
        i = Class_id//cols
        j = Class_id%cols
        
        ax = axs[i, j]
        
        f1 = lambda x: '%.2f%%' % (x * 100)
        num_rows = len(data)
        average_accuracy = f1(accuracy_score(data['True_labels'], data['Predicted_labels']))
        
        ax.set_title(f"{Class_type[Class_id]} \nAverage accuracy:{average_accuracy} \n({num_rows} pts)", fontsize=15, fontweight = 'bold')  # 添加小标题
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize = 15)
        #ax.set_xlabel('Predicted source', fontsize = 30)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize = 15)
        #ax.set_ylabel('True source', fontsize = 30)
        
        f2 = lambda x: '%.1f%%' % (x * 100)        
        for x in range(len(labels)):
            for y in range(len(labels)):
                if x == y:
                    text = ax.text(y, x, f2(con_matrix_normalized[x, y]) + "\n" + "(" + str(con_matrix[x, y]) + ")", ha="center", va="center", color="b", fontsize = 8)
                else:
                    text = ax.text(y, x, f2(con_matrix_normalized[x, y]) + "\n" + "(" + str(con_matrix[x, y]) + ")", ha="center", va="center", color="w", fontsize = 8)

        if len(set(con_matrix_normalized.flatten())) == 1:
            cmap = ListedColormap(['#fde724'])
            ax.imshow(con_matrix_normalized, cmap=cmap)
        else:
            ax.imshow(con_matrix_normalized)
        
        if Class_id == 0: 
            im = ax.imshow(con_matrix_normalized)
    

    for idx, ax in enumerate(axs.flatten()):
        if idx >= len(Class_type):
            ax.axis('off')
    
    fig.subplots_adjust(hspace=0.6, wspace=0.02)
    cbar_ax = fig.add_axes([0.62, 0.15, 0.25, 0.02])
    fig.colorbar(im, cax=cbar_ax, ticks=colorbar_ticks, orientation='horizontal')
    cbar_ax.set_xticklabels(['{:.0%}'.format(x) for x in colorbar_ticks], fontsize=22)
    
    fig.text(0.5, 0.09, 'Predicted labels', ha='center', fontsize=30, fontweight = 'bold')

    fig.text(0.09, 0.5, 'True labels', va='center', rotation='vertical', fontsize=30, fontweight = 'bold')
    
    #plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=500, bbox_inches='tight') # 通过 bbox_inches 参数确保保存的图像包含所有的绘图元素
    plt.show()
    
con_matrix_norm_sub_plot(Dataframe, fig_path = '../Out put/Class_confusion_matrix.pdf')


Class_type = ["Si-Al", "Si-rich","Si-Al-Ca","Si-Al-Fe","Si-Al-K-Mg","Si-Al-Mg","Si-Al-Na","Si-Al-Ca-Mg-Fe","Si-Al-Ca-Mg","Si-Ca","Al-rich","Al-Ca","Ca-Mg","Ca-Mg-Cl","P-containing","K-containing","Fly ash","C-rich","Fe-rich","Ca-rich","S-Ca","Ti-rich","Fe-Ca-S","Fe-K-Cl", "Mn-rich","K-Cl","Pb-Cl","Pb-Fe-Cl","Pb-Cl-K","S-Al-Ca-Mg","S-Ca-Pb","S-Ca-K-Cl","Other"]
accuracy = dict()
f1 = lambda x: '%.2f%%' % (x * 100)

for Class_id in range(len(Class_type)):       
    data = Dataframe[Dataframe['Class'] == Class_type[Class_id]]
    accuracy[Class_type[Class_id]] = accuracy_score(data['True_labels'], data['Predicted_labels'])
print(accuracy)

my_list = list(accuracy.items())
df = pd.DataFrame(my_list, columns=["Class", "accuracy"])
print(df)

df.to_excel("../Out put/Class_accuracy.xlsx", index=False)

report = classification_report(y_test_le, y_pred, digits=3)
print(report)

y_test_bin = label_binarize(y_test_le, classes = [0, 1, 2, 3, 4, 5])
aupr_per_class = average_precision_score(y_test_bin, y_pred_prob, average=None)
print(f"AUPR per class: {aupr_per_class}")

