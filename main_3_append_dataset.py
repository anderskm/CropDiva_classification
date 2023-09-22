import glob
import hashlib
import json
import os
import numpy as np
import pandas as pd
import pickle
import shutil
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tqdm

dataset_folder = 'Datasets'
parent_dataset = '4a64dfabd48e2cf59ba9cc59'
inliners_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_NormCW_less_dataAug__0e244979/inliers__net_0e244979_train_4a64dfabd48e2cf59ba9cc59_pred_87bb031e36e9d24636e9d246.pkl'
inliner_image_folder = 'D:/CropDiva_tmp/05SEP23_NotApproved_ScaledPadding'
parent_image_folder = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/TrainingData/05SEP23_scale224px_50px'
output_image_folder = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/TrainingData/05SEP23_scale224px_50px__merged'

df_train = pd.read_pickle(os.path.join(dataset_folder, parent_dataset, 'dataframe_annotations_' + parent_dataset + '_Train.pkl'))
df_test = pd.read_pickle(os.path.join(dataset_folder, parent_dataset, 'dataframe_annotations_' + parent_dataset + '_Test.pkl'))
df_validation = pd.read_pickle(os.path.join(dataset_folder, parent_dataset, 'dataframe_annotations_' + parent_dataset + '_Validation.pkl'))

df_inliers = pd.read_pickle(inliners_path)

print(df_inliers.head())

# Load labels dict
labels_dict_file = glob.glob(os.path.join(dataset_folder, parent_dataset,'labels_dict_' + parent_dataset + '.json'))[0]
with open(labels_dict_file,'r') as fob:
    labels_dict = json.load(fob)
N_classes = len(labels_dict)
label_no_2_label = dict([(value, key) for key, value in labels_dict.items()])

# Update label no and names
df_inliers['label_no'] = df_inliers['pred_label_no']
df_inliers['label'] = [label_no_2_label[lo] for lo in df_inliers['label_no'].values.tolist()] #df_inliers[['label_no']].apply(lambda x: label_no_2_label[np.asarray(x).reshape(-1,1)], axis=1)

# Update one-hot encoding
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.asarray(df_train['label_no']).reshape(-1,1))
df_inliers['label_one_hot'] = df_inliers[['label_no']].apply(lambda x: ohe.transform(np.asarray(x).reshape(-1,1)), axis=1)


le_file = glob.glob(os.path.join('Datasets',parent_dataset, 'label_encoder_*.pkl'))[0]
print(le_file)
with open(le_file, 'rb') as fob:
    le, label_dict = pickle.load(fob)


for row in tqdm.tqdm(df_inliers.iterrows(), total=df_inliers.shape[0]):
    src = os.path.join(inliner_image_folder, row[1]['folder'], row[1]['image'])
    dst = os.path.join(output_image_folder, row[1]['label'], row[1]['image'])
    shutil.copyfile(src, dst)

df_inliers['folder'] = df_inliers['label']

# Remove extra columns before concatenation
df_inliers = df_inliers.drop(['predictions', 'pred_label_no'], axis=1)

print(df_inliers)

print('Class distribution BEFORE including new samples:')
df_all = pd.concat([df_train, df_validation, df_test])
print(df_all.groupby(['Dataset','label'])['label'].count().unstack())

# Concat inliers to training data
df_train = pd.concat([df_train, df_inliers])


print('Class distribution AFTER including new samples:')
df_all = pd.concat([df_train, df_validation, df_test])
print(df_all.groupby(['Dataset','label'])['label'].count().unstack())

# Create unique identifier from the cluster dataset assignment
hash_func_train = hashlib.blake2s(digest_size=4)
hash_func_train.update(bytes(''.join([str(c) for c in df_train['ImageID'].unique()]), 'utf-8'))
hash_func_validation = hashlib.blake2s(digest_size=4)
hash_func_validation.update(bytes(''.join([str(c) for c in df_validation['ImageID'].unique()]), 'utf-8'))
hash_func_test = hashlib.blake2s(digest_size=4)
hash_func_test.update(bytes(''.join([str(c) for c in df_test['ImageID'].unique()]), 'utf-8'))

dataset_split_identifier = hash_func_train.hexdigest() + hash_func_validation.hexdigest() + hash_func_test.hexdigest()
print('Dataset identifier:', dataset_split_identifier)

# dataset_split_identifier = 'akm_test_123'

# Create output folder
output_folder = os.path.join('Datasets', dataset_split_identifier)
os.makedirs(output_folder, exist_ok=False)
print('Dumping to dataset folder: ', output_folder)

with open(os.path.join(output_folder, 'inheritence_' + dataset_split_identifier +'.json'),'w') as fob:
    json.dump({'dataset_folder': dataset_folder,
               'parent_dataset': parent_dataset,
               'inliners_path': inliners_path,
               'inliner_image_folder': inliner_image_folder,
               'parent_image_folder': parent_image_folder,
               'output_image_folder': output_image_folder
               },
               fob,
               indent=3)

# Dump labels to label no in json format
label_dict_out_path = os.path.join(output_folder, 'labels_dict_' + dataset_split_identifier +'.json')
print('labels dict: ', label_dict_out_path)
with open(label_dict_out_path,'w') as fob:
    label_dict = dict(zip(list(le.inverse_transform([i for i in range(N_classes)])), [i for i in range(N_classes)]))
    json.dump(label_dict, fob)
label_encoder_path = os.path.join(output_folder, 'label_encoder_' + dataset_split_identifier + '.pkl')
print('label encoder: ', label_encoder_path)
with open(label_encoder_path,mode='wb') as fob:
    pickle.dump((le, label_dict), fob)

# Save datasets w. unique identifier
train_set_path = os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Train.pkl')
print('Train set: ', train_set_path)
df_train.to_pickle(train_set_path)
val_set_path = os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Validation.pkl')
print('Validation set: ', val_set_path)
df_validation.to_pickle(val_set_path)
test_set_path = os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Test.pkl')
print('Test set: ', test_set_path)
df_test.to_pickle(test_set_path)

print('done')