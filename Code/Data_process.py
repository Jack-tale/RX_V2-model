# ## Processing of EDS and Particle Size Data

import pandas as pd
import numpy as np
import pickle
import zipfile
import os
import glob
from sklearn.model_selection import train_test_split
import imageio
import time
import random
import sys
import copy
import json
from PIL import Image
import shutil

zip_file = 'Row data.zip'
current_path = os.getcwd()
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(current_path)
print("Unzip finish.")


data = pd.read_excel("../Row data/6_source_data.xlsx", sheet_name = 0)
data.insert(0, 'id_code', list(range(len(data)))) # Insert a column for all samples

for index, value in enumerate(data.columns):
    print(index, value)


# Divide the data into training sets and test sets
dataset = np.array(data)
labels = np.array(data.iloc[:,38])
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, test_size = .1, stratify = labels, random_state = 1)

# Sort the training and test set data by id_code to facilitate mapping with image labels
train_data = pd.DataFrame(train_dataset)
train_data.columns = data.columns
train_data = train_data.sort_values(by='id_code')

test_data = pd.DataFrame(test_dataset)
test_data.columns = data.columns
test_data = test_data.sort_values(by='id_code')

# Save the dataset
writer = pd.ExcelWriter('../Row data/train_test_row_data.xlsx')
train_data.to_excel(writer, sheet_name = 'train_data', index = False)
test_data.to_excel(writer, sheet_name = 'test_data', index = False)
writer._save()


# The dataset (shape parameter + particle size + element) is used for multi-model comparison
train_data_1 = np.array(train_data.iloc[:,list([2])+list(range(6,36))])
test_data_1 = np.array(test_data.iloc[:,list([2])+list(range(6,36))])
train_labels_1 = np.array(train_data.iloc[:,38])
test_labels_1 = np.array(test_data.iloc[:,38])

# Save the data
pickle_file = '../Row data/train_test_data_1.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file ...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset':train_data_1,
                    'train_labels': train_labels_1,
                    'test_dataset': test_data_1,
                    'test_labels': test_labels_1
                },
                pfile, pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
print('Data cached in pickle file.')

# The dataset (particle size + elements) is used to train the RX model
train_data_2 = np.array(train_data.iloc[:,list(range(6,11))+list(range(13,36))])
test_data_2 = np.array(test_data.iloc[:,list(range(6,11))+list(range(13,36))])
train_labels_2 = np.array(train_data.iloc[:,38])
test_labels_2 = np.array(test_data.iloc[:,38])


# Save the data
pickle_file = '../Row data/train_test_data_2.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file ...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset':train_data_2,
                    'train_labels': train_labels_2,
                    'test_dataset': test_data_2,
                    'test_labels': test_labels_2
                },
                pfile, pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
print('Data cached in pickle file.')


# ## 2. Organizing Microscopic Images

path = '../Row data/source_pic_data'
files = os.listdir(path)


# Consolidating images into one folder and renaming them to correspond with predefined particle names in the Excel file
for file_folder in files:
    
    Summarize_pic_folder = '../Row data'

    if not os.path.exists(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder')):
        os.makedirs(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder'))
    
    #for file_index in os.listdir(os.path.join(path, file_folder)):
    file_index = "01"
    file_subfolder = os.listdir(os.path.join(path, file_folder, file_index))

    none_xlsx_file_subfolder = [file for file in file_subfolder if not file.endswith('.xlsx')]
    for pic_folder_name in none_xlsx_file_subfolder:
        pic_folder = os.path.join(path, file_folder, file_index, pic_folder_name)

        for image in os.listdir(pic_folder):
            source_path = os.path.join(pic_folder, image)
            destination_name = os.path.join(os.path.join(Summarize_pic_folder, 'Summarize_pic_folder'), f"{file_folder}_{pic_folder_name}_{image[:-4].zfill(5)}"+".png") # 图片目标路径
            shutil.copyfile(source_path, destination_name)


train_data, test_data = train_data[['Part #', 'file_name', 'Source']], test_data[['Part #', 'file_name', 'Source']]

files = os.listdir(path) # ['Biomass', 'Coal', 'Construction', 'Road dust', 'Soil', 'Steel']
folder_list = [str(i) for i in range(len(files))]
source_dict = {k: v for k, v in zip(folder_list, files)}
print(source_dict)


for key, source_label in source_dict.items():

    train_folder = '../Row data/Training data/train'
    test_folder = '../Row data/Training data/test'
    if not os.path.exists(os.path.join(train_folder, key)):
        os.makedirs(os.path.join(train_folder, key))
    if not os.path.exists(os.path.join(test_folder, key)):
        os.makedirs(os.path.join(test_folder, key))
        
    # train pic
    for index, row in train_data[train_data['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(Summarize_pic_folder, 'Summarize_pic_folder', img_name + '.png')
        dst_path = os.path.join(train_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)  
    
    # test pic
    for index, row in test_data[test_data['Source'] == source_label].iterrows():
        img_name = source_label + '_' + row['file_name'] + '_' + str(row['Part #']).zfill(5)
        src_path = os.path.join(Summarize_pic_folder, 'Summarize_pic_folder', img_name + '.png')
        dst_path = os.path.join(test_folder, key, img_name + '.png')
        shutil.copyfile(src_path, dst_path)

shutil.rmtree('../Row data/Summarize_pic_folder')

