# import argparse
# from datetime import datetime
# import glob
import hashlib
import json
import numpy as np
import os
# import matplotlib.pyplot as plt
import pickle
# import plotly
# import plotly.express as px
import pandas as pd
import random
# import scipy.spatial.distance
from scipy.stats import chisquare
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tqdm

import utils

input_dataframe_filename = 'dataframe_annotations__filtered.pkl'
ratios = [0.7, 0.15, 0.15] # Train, validation, test
load_rng_from_identifier = '5e38f04b85438b1289928e61' # Set to identifier to load previous used rng_state. if set to '', use a new random seed/state set by the random number generators
load_rng_from_identifier = '' # Set to identifier to load previous used rng_state. if set to '', use a new random seed/state set by the random number generators

# Load dataframe
df = pd.read_pickle(input_dataframe_filename)

# Get state of random number generator. Dump to file later, such that dataset can be recreate using same state if needed.
state_py = random.getstate()
state_np = np.random.get_state()

# Load previous rng_state
if load_rng_from_identifier:
    fob = open(os.path.join('Datasets', load_rng_from_identifier, 'rng_state_' + load_rng_from_identifier + '.pkl'), 'rb')
    state_py, state_np = pickle.load(fob)
    fob.close()

# Set rng states
random.setstate(state_py)
np.random.set_state(state_np)

# Set class label no + one-hot-encoding to make sure that it is consistent across the three datasets
N_labels = len(df['label'].unique())
le = LabelEncoder()
df['label_no'] = le.fit_transform(df['label'])
ohe = OneHotEncoder(sparse=False)
ohe.fit(np.asarray(df['label_no']).reshape(-1,1))
df['label_one_hot'] = df[['label_no']].apply(lambda x: ohe.transform(np.asarray(x).reshape(-1,1)), axis=1)

images_per_label = np.asarray(df.groupby(['label'])['label'].count())
# cluster_weights = np.asarray(df.groupby(['cluster','label'])['label'].count().unstack().fillna(0))
df_cluster_weights = df.groupby(['ImageID','label'])['label'].count().unstack().fillna(0)

training_clusters = []
training_cluster_weights = np.zeros(images_per_label.shape)
validation_clusters = []
validation_cluster_weights = np.zeros(images_per_label.shape)
test_clusters = []
test_cluster_weights = np.zeros(images_per_label.shape)

chisq_tr_prev = np.Inf
chisq_va_prev = np.Inf
chisq_te_prev = np.Inf

# Scramble cluster order, such that they are added in random order
clusters_random_order = np.random.permutation(df['ImageID'].unique())

for cluster in tqdm.tqdm(clusters_random_order, desc='Assigning Images to datasets (' + ','.join([str(r) for r in ratios]) + '): '):
    cluster_weights = df_cluster_weights.iloc[df_cluster_weights.index == cluster,:].values.squeeze()
    # Get chi-squared value of adding cluster to each of the datasets
    chisq_tr, p_tr = chisquare(training_cluster_weights + cluster_weights, np.round(images_per_label*ratios[0]), ddof=len(images_per_label)-1)
    chisq_va, p_va = chisquare(validation_cluster_weights + cluster_weights, np.round(images_per_label*ratios[1]), ddof=len(images_per_label)-1)
    chisq_te, p_te = chisquare(test_cluster_weights + cluster_weights, np.round(images_per_label*ratios[2]), ddof=len(images_per_label)-1)
    
    # Get decrease in chi-squared value for adding cluster to each of the datasets
    delta_chi_tr = chisq_tr_prev - chisq_tr
    delta_chi_va = chisq_va_prev - chisq_va
    delta_chi_te = chisq_te_prev - chisq_te
    
    # Add cluster to dataset with largest decrease in chi-squared value
    if (delta_chi_te > 0) & (delta_chi_te >= delta_chi_tr) & (delta_chi_te >= delta_chi_va):
        test_clusters.append(cluster)
        test_cluster_weights += cluster_weights
        chisq_te_prev = chisq_te
    elif (delta_chi_va > 0) & (delta_chi_va > delta_chi_tr) & (delta_chi_va > delta_chi_te):
        validation_clusters.append(cluster)
        validation_cluster_weights += cluster_weights
        chisq_va_prev = chisq_va
    else:
        training_clusters.append(cluster)
        training_cluster_weights += cluster_weights
        chisq_tr_prev = chisq_tr

# Calculate chi-squared value and p-value of final cluster distribution among train, validation and test set
chisq_tr, p_tr = chisquare(training_cluster_weights, np.round(images_per_label*ratios[0]), ddof=len(images_per_label)-1)
chisq_va, p_va = chisquare(validation_cluster_weights, np.round(images_per_label*ratios[1]), ddof=len(images_per_label)-1)
chisq_te, p_te = chisquare(test_cluster_weights, np.round(images_per_label*ratios[2]), ddof=len(images_per_label)-1)

# Store assigned dataset to dataframe
df['Dataset'] = ''
df.loc[df['ImageID'].isin(training_clusters),'Dataset'] = 'Train'
df.loc[df['ImageID'].isin(validation_clusters),'Dataset'] = 'Validation'
df.loc[df['ImageID'].isin(test_clusters),'Dataset'] = 'Test'

# Print overview of distributions
print('Labels per dataset')
df_datasets_overview = df.groupby(['Dataset','label'])['label'].count().unstack()
print(df_datasets_overview)
df_datasets_ratios = df_datasets_overview / df_datasets_overview.sum()
df_datasets_ratios['TARGET'] = [ratios[2], ratios[0], ratios[1]] # Specified order is different from alphabetic
df_datasets_ratios['Chi_squared'] = [chisq_te, chisq_tr, chisq_va]
df_datasets_ratios['P-value'] = [p_te, p_tr, p_va]
print(df_datasets_ratios)

# Create unique identifier from the cluster dataset assignment
hash_func_train = hashlib.blake2s(digest_size=4)
hash_func_train.update(bytes(''.join([str(c) for c in training_clusters]), 'utf-8'))
hash_func_validation = hashlib.blake2s(digest_size=4)
hash_func_validation.update(bytes(''.join([str(c) for c in validation_clusters]), 'utf-8'))
hash_func_test = hashlib.blake2s(digest_size=4)
hash_func_test.update(bytes(''.join([str(c) for c in test_clusters]), 'utf-8'))

dataset_split_identifier = hash_func_train.hexdigest() + hash_func_validation.hexdigest() + hash_func_test.hexdigest()
print('Dataset identifier:')
print(dataset_split_identifier)

# Create output folder
output_folder = os.path.join('Datasets', dataset_split_identifier)
os.makedirs(output_folder, exist_ok=False)

# Dump state of random number generator prior to splitting dataset
fob = open(os.path.join(output_folder, 'rng_state_' + dataset_split_identifier + '.pkl'), mode='wb')
pickle.dump((state_py, state_np), fob)
fob.close()

# Dump labels to label no in json format
fob = open(os.path.join(output_folder, 'labels_dict_' + dataset_split_identifier +'.json'),'w')
label_dict = dict(zip(list(le.inverse_transform([i for i in range(N_labels)])), [i for i in range(N_labels)]))
json.dump(label_dict, fob)
fob = open(os.path.join(output_folder, 'label_encoder_' + dataset_split_identifier + '.pkl'),mode='wb')
pickle.dump((le, label_dict), fob)
fob.close()

# # Plot image locations on map. Color by assigned dataset
# fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='Dataset', mapbox_style='carto-positron')
# fig1.show()
# plotly.offline.plot(fig1, filename=os.path.join(output_folder, 'map_of_datasets_' + dataset_split_identifier + '.html'))

# Create dataframe for each dataset
df_train,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Train')
df_validation,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Validation')
df_test,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Test')

# Save datasets w. unique identifier
df_train.to_pickle(os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Train.pkl'))
df_validation.to_pickle(os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Validation.pkl'))
df_test.to_pickle(os.path.join(output_folder, 'dataframe_annotations_' + dataset_split_identifier + '_Test.pkl'))

print('done')