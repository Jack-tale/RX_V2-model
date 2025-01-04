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



train_data = pd.read_excel("../Row data/train_test_row_data.xlsx", sheet_name = 0)
for index, value in enumerate(train_data.columns):
    print(index, value)
dataset = np.array(train_data)
labels = np.array(train_data.iloc[:,38])

# Divide the dataset into training subset and validation subset.
train_subset, validation_subset, train_subset_labels, validation_subset_labels = train_test_split(dataset, labels, test_size = .11, stratify = labels)

# Sorting train_subset and validation_subset by id_code
train_subset = pd.DataFrame(train_subset)
train_subset.columns = train_data.columns
train_subset = train_subset.sort_values(by='id_code')
validation_subset = pd.DataFrame(validation_subset)
validation_subset.columns = train_data.columns
validation_subset = validation_subset.sort_values(by='id_code')


# Merge the images from the path "../Row data/Training data/train" into another folder.

source_dir = '../Row data/Training data/train'
target_dir = '../Row data/Training data/temp'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)
    for file in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file)
        shutil.copy(file_path, target_dir)

files = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']
folder_list = [str(i) for i in range(len(files))]
source_dict = {k: v for k, v in zip(folder_list, files)}
source_dict

for key, source_label in source_dict.items():
    train_subset_folder = '../Row data/Training data/train_subset'
    validation_subset_folder = '../Row data/Training data/validation_subset'
    if not os.path.exists(os.path.join(train_subset_folder, key)):
        os.makedirs(os.path.join(train_subset_folder, key))
    if not os.path.exists(os.path.join(validation_subset_folder, key)):
        os.makedirs(os.path.join(validation_subset_folder, key))
    # train_subset pic
    for index, row in train_subset[train_subset['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(target_dir, img_name + '.png')
        dst_path = os.path.join(train_subset_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)  
    # validation_subset pic
    for index, row in validation_subset[validation_subset['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(target_dir, img_name + '.png')
        dst_path = os.path.join(validation_subset_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)
shutil.rmtree("../Row data/Training data/temp")




# ## ResNet model training

data_dir = '../Row data/Training data/'
train_dir = data_dir + 'train_subset/'
test_dir = data_dir + 'validation_subset/'

data_transforms = {
    'train_subset': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(90),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation_subset': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train_subset', 'validation_subset']}




# Selecting the appropriate number of processes for data loading, using the test set data as an example
import time

num_workers_list = [10, 11, 12]
batch_size = 16
epochs = 3
sample_count = len(image_datasets['validation_subset'])

for num_workers in num_workers_list:
    dataloader = torch.utils.data.DataLoader(dataset = image_datasets['validation_subset'], batch_size = batch_size, shuffle = True, num_workers=num_workers)
    start_time = time.time()
    for _ in range(epochs):
        for _ in dataloader:
            pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"num_workers={num_workers}: total time={elapsed_time}, time per epoch={(elapsed_time/epochs)}, time per sample={(elapsed_time/sample_count)}")



# Viewing Images

batch_size = 16
num_workers = 12

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = True, num_workers=num_workers) for x in ['train_subset', 'validation_subset']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train_subset', 'validation_subset']}
class_names = image_datasets['train_subset'].classes
files = ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']
folder_list = [str(i) for i in range(6)]
source_dict = {k: v for k, v in zip(folder_list, files)}

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze() 
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2
dataiter = iter(dataloaders['validation_subset'])
inputs, classes = next(dataiter)

for idx in range (min(columns*rows, len(inputs))):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(inputs[idx]))
    ax.set_title(source_dict[str(int(class_names[classes[idx]]))])
plt.show()


# Model training

def initialize_model(num_classes, use_pretrained = True):
    model_ft = models.resnet152(pretrained = use_pretrained)
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                nn.LogSoftmax(dim = 1))
    
    return model_ft

model_ft = initialize_model(6, use_pretrained=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_ft = model_ft.to(device)

print("Params to learn:")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t", name)



# Optimizer Configuration
optimizer_ft = optim.Adam(params_to_update, lr = 1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)
criterion = nn.NLLLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs = 25, filename = 'checkpoint.pth'):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)
    
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    test_losses = []
    LRs = [optimizer.param_groups[0]['lr']] 
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        for phase in ['train_subset', 'validation_subset']:
            if phase == 'train_subset':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterating through the data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Clearing gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train_subset'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) #Finding the maximum value and its corresponding index in the predicted result
                    
                    if phase == 'train_subset':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            
            if phase == 'validation_subset' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(state, filename)
            if phase == 'validation_subset':
                val_acc_history.append(epoch_acc)
                test_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'validation_subset':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, test_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, test_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs = 30)
torch.save(model_ft, "../Trained model/ResNet_V2.pth")
# The "ResNet_V2.pth" file is too large to upload to GitHub, so it has been uploaded to the Figshare platform (https://doi.org/10.6084/m9.figshare.28137509.v1).




# performance on test dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load("../Trained model/ResNet.pth")
model = model_ft
model.to(device)
model.eval()

data_dir = '../Row data/Training data/'
test_dir = data_dir + 'test/'

val_tf = transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(test_dir, val_tf)
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = 64, num_workers = 11)

running_loss = 0.0
running_corrects = 0

y_true = []
y_pred = []
df = pd.DataFrame()

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    _, preds = torch.max(outputs, 1)
    probs = nn.functional.softmax(outputs, dim=1)
    class_probabilities = probs.detach().cpu().numpy()

    y_true.extend(labels.detach().cpu().numpy())
    y_pred.extend(preds.detach().cpu().numpy())
    df = df._append(pd.DataFrame(class_probabilities), ignore_index=True)

report = classification_report(y_true, y_pred, digits=3)
print(report)



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

