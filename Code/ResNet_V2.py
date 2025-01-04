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
import gc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import warnings

# During the ResNet training phase, the training strategy uses a training set: validation set: test set ratio of 8:1:1.

train_data = pd.read_excel("../Row data/train_test_row_data.xlsx", sheet_name = 0)
for index, value in enumerate(train_data.columns):
    print(index, value)
dataset = np.array(train_data)
labels = np.array(train_data.iloc[:,38])

# Divide the dataset into training subset and validation subset (8:1).
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

