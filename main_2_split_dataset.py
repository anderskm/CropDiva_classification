import argparse
# from datetime import datetime
import glob
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
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tqdm

import utils

def main():    # Setup input argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_df', action='store', default='dataframe_annotations__all__filtered3.pkl', type=str, help='Filename of pickle file containing filtered dataframe (default: %(default)s).')
    parser.add_argument('--ratios', action='store', nargs=3, default=[0.7, 0.15, 0.15], help="3 floats indication the splits between training, validation and test set (default: %(default)s).")
    parser.add_argument('--parent_dataset', action='store', default='', type=str, help='Path to dataset folder of dataset from which labels should be inherited (default: %(default)s).')
    parser.add_argument('--labels_dict', action='store', default='', type=str, help='Filename of json file with dictionany of classes. Use classes if extra classes than in parent dataset (default: %(default)s).')
    parser.add_argument('--load_rng_from_identifier', action='store', default='', type=str, help='Set to identifier to load previous used rng_state. if set to '', use a new random seed/state set by the random number generators')

    args = vars(parser.parse_known_args()[0])
    print('\nArguments: ', args)

    input_dataframe_filename = args['input_df']
    ratios = [float(i) for i in args['ratios']]
    load_rng_from_identifier = args['load_rng_from_identifier']
    parent_dataset = args['parent_dataset']
    labels_dict_file = args['labels_dict']

    # input_dataframe_filename = 'dataframe_annotations__filtered.pkl'
    # ratios = [0.7, 0.15, 0.15] # Train, validation, test
    # load_rng_from_identifier = '5e38f04b85438b1289928e61' # Set to identifier to load previous used rng_state. if set to '', use a new random seed/state set by the random number generators
    # load_rng_from_identifier = '' # Set to identifier to load previous used rng_state. if set to '', use a new random seed/state set by the random number generators

    # Load dataframe
    df = pd.read_pickle(input_dataframe_filename)

    # Get state of random number generator. Dump to file later, such that dataset can be recreate using same state if needed.
    state_py = random.getstate()
    state_np = np.random.get_state()

    # Load previous rng_state
    if load_rng_from_identifier:
        with open(os.path.join('Datasets', load_rng_from_identifier, 'rng_state_' + load_rng_from_identifier + '.pkl'), 'rb') as fob:
            state_py, state_np = pickle.load(fob)

    # Set rng states
    random.setstate(state_py)
    np.random.set_state(state_np)

    # Set class label no + one-hot-encoding to make sure that it is consistent across the three datasets
    N_labels = len(df['label'].unique())
    if parent_dataset:
        #fob = open(os.path.join('Datasets', 'label_encoder_' + dataset_split_identifier + '.pkl'),mode='wb')
        print('Reusing existing label encoder from:')
        le_file = glob.glob(os.path.join('Datasets',parent_dataset, 'label_encoder_*.pkl'))[0]
        print(le_file)
        with open(le_file, 'rb') as fob:
            le, labels_dict = pickle.load(fob)
        if labels_dict_file:
            # Update label encoder to include new labels from labels dict
            with open(labels_dict_file,'r') as fob:
                labels_dict = json.load(fob)
            new_labels = [k for k in labels_dict.keys() if k not in le.classes_]
            for l in new_labels:
                le.classes_ = np.append(le.classes_, l)
        df['label_no'] = le.transform(df['label'])
    else:
        N_labels = len(df['label'].unique())
        le = LabelEncoder()
        df['label_no'] = le.fit_transform(df['label'])
    print('Creating one-hot encoding:')
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(np.asarray(df['label_no']).reshape(-1,1))
    df['label_one_hot'] = df[['label_no']].apply(lambda x: ohe.transform(np.asarray(x).reshape(-1,1)), axis=1)

    print('Preparing to assign images to datasets...')
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
    if test_cluster_weights.sum() == 0.0:
        df_datasets_ratios = pd.concat([pd.DataFrame(data=np.zeros((1,len(df_datasets_ratios.columns))),
                                                    index=pd.Index(['Test'],name='Dataset'),
                                                    columns=df_datasets_ratios.columns),
                                       df_datasets_ratios])
    if validation_cluster_weights.sum() == 0.0:
        df_datasets_ratios = pd.concat([df_datasets_ratios,
                                        pd.DataFrame(data=np.zeros((1,len(df_datasets_ratios.columns))),
                                                    index=pd.Index(['Validation'],name='Dataset'),
                                                    columns=df_datasets_ratios.columns)])
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
    print('Dataset identifier:', dataset_split_identifier)

    # Create output folder
    output_folder = os.path.join('Datasets', dataset_split_identifier)
    os.makedirs(output_folder, exist_ok=False)
    print('Dumping to dataset folder: ', output_folder)


    args_path = os.path.join(output_folder, 'args_' + dataset_split_identifier +'.json')
    print('args file: ', args_path)
    with open(args_path,'w') as fob:
        json.dump(args, fob, indent=3)

    # Dump state of random number generator prior to splitting dataset
    rng_state_path = os.path.join(output_folder, 'rng_state_' + dataset_split_identifier + '.pkl')
    print('RNG state file: ', rng_state_path)
    with open(rng_state_path, mode='wb') as fob:
        pickle.dump((state_py, state_np), fob)

    # Dump labels to label no in json format
    labels_dict_out_path = os.path.join(output_folder, 'labels_dict_' + dataset_split_identifier +'.json')
    print('labels dict: ', labels_dict_out_path)
    with open(labels_dict_out_path,'w') as fob:
        labels_dict = dict(zip(list(le.inverse_transform([i for i in range(N_labels)])), [i for i in range(N_labels)]))
        json.dump(labels_dict, fob)
    label_encoder_path = os.path.join(output_folder, 'label_encoder_' + dataset_split_identifier + '.pkl')
    print('label encoder: ', label_encoder_path)
    with open(label_encoder_path,mode='wb') as fob:
        pickle.dump((le, labels_dict), fob)

    # Create dataframe for each dataset
    df_train,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Train')
    df_validation,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Validation')
    df_test,_ = utils.dataframe_filtering(df, df['Dataset'] == 'Test')

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

    print('Dataset identifier: ', dataset_split_identifier)

    print('done')

if __name__ == '__main__':
    main()
