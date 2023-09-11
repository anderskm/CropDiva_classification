import copy
import confusionmatrix
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tqdm

networks_main_folder = 'C:/Vejdirektorat/vjd-invasive-plantearter/networks_weights_pooling_simple'

dataset_folder = 'datasets'
network_folders = glob.glob(os.path.join(networks_main_folder, '*'))

print(network_folders)

DFs = []
for network_folder in tqdm.tqdm(network_folders, desc='Loading networks (conf. and history)'):
    if os.path.isdir(network_folder):
        _, network_name = os.path.split(network_folder)

        # Load configuration
        with open(os.path.join(network_folder, 'configuration.json'),'r') as fob:
            configuration = json.load(fob)

        # Load training log
        with open(os.path.join(network_folder, 'train_history.json'), 'r') as fob:
            train_history = json.load(fob)

        # Load labels
        labels_dict_file = os.path.join(dataset_folder, configuration['dataset_id'],'labels_dict_' + configuration['dataset_id'] + '.json')
        with open(labels_dict_file,'r') as fob:
            labels_dict = json.load(fob)
        N_classes = len(labels_dict)

        # Load conf mat
        CMat_best = confusionmatrix.confusionmatrix(N_classes, labels=[key for key in labels_dict.keys()])
        CMat_best.Load(os.path.join(network_folder, 'ConfMat_best_validation.csv'))

        labels_simple = ['Invasive','None']
        CMat_best_simple = CMat_best.MergeLabels([[0, 1, 2, 3, 5, 6], [4]], new_labels=labels_simple)

        # Store in dataframe
        val_max_idx = np.argmax(train_history['history']['val_accuracy'])
        column_names = [key for key in configuration.keys()]
        column_names += ['epoch']
        column_names += [key for key in train_history['history'].keys()]
        column_names += ['cmat_best', 'cmat_last', 'cmat_best_simple']
        df_network = pd.DataFrame(columns=column_names)
        for e, epoch in enumerate(train_history['epoch']):
            epoch_dict = copy.copy(configuration)
            epoch_dict['epoch'] = epoch
            for key in train_history['history'].keys():
                epoch_dict[key] = train_history['history'][key][e]
            if e == val_max_idx:
                epoch_dict['cmat_best'] = CMat_best
                epoch_dict['cmat_best_simple'] = CMat_best_simple
            else:
                epoch_dict['cmat_best'] = confusionmatrix.confusionmatrix(N_classes, labels=[key for key in labels_dict.keys()])
                epoch_dict['cmat_best_simple'] = confusionmatrix.confusionmatrix(2, labels=labels_simple)
            df_network = df_network.append(epoch_dict, ignore_index=True)
        DFs.append(df_network)

# Concatenate all network dataframes
df_all = pd.concat(DFs, ignore_index=True)

df_all['image_height'] = df_all[['image_size']].apply(lambda x: int(x[0][0]), axis=1)
df_all['image_width']  = df_all[['image_size']].apply(lambda x: int(x[0][1]), axis=1)

# df_best = df_all.groupby(['weights','pooling','image_width'])['val_accuracy'].max().reset_index()
idx_best = df_all.groupby(['weights','pooling','image_width'])['val_accuracy'].transform('max') == df_all['val_accuracy']
df_best = df_all[idx_best]

ax = sns.lineplot(data=df_best, x='image_width', y='val_accuracy', hue='weights', style='pooling', markers=True)
ax.set(xscale="log")
ax.set(xticks=df_best['image_width'].unique())
ax.set(xticklabels=df_best['image_width'].unique())
plt.show()

df_best['best_accuracy'] = df_best[['cmat_best']].applymap(lambda x: x.accuracy())
df_best['best_precision'] = df_best[['cmat_best']].applymap(lambda x: x.precision()[0])
df_best['best_recall'] = df_best[['cmat_best']].applymap(lambda x: x.recall()[0])
df_best['best_f1'] = df_best[['cmat_best']].applymap(lambda x: x.fScore()[0])

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Invasive species')
axes = axes.flatten()
sns.lineplot(ax=axes[0], data=df_best, x='image_width', y='best_accuracy', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[1], data=df_best, x='image_width', y='best_f1', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[2], data=df_best, x='image_width', y='best_precision', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[3], data=df_best, x='image_width', y='best_recall', hue='weights', style='pooling', markers=True)
for ax in axes:
    ax.set(xscale="log")
    ax.set(xticks=df_best['image_width'].unique())
    ax.set(xticklabels=df_best['image_width'].unique())
plt.show()

df_best['simple_accuracy'] = df_best[['cmat_best_simple']].applymap(lambda x: x.accuracy())
df_best['simple_precision'] = df_best[['cmat_best_simple']].applymap(lambda x: x.precision()[0])
df_best['simple_recall'] = df_best[['cmat_best_simple']].applymap(lambda x: x.recall()[0])
df_best['simple_f1'] = df_best[['cmat_best_simple']].applymap(lambda x: x.fScore()[0])

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Invasive vs. none-invasive')
axes = axes.flatten()
sns.lineplot(ax=axes[0], data=df_best[(df_best['weights'] == 'random') & (df_best['pooling'] == 'max')], x='image_width', y='simple_accuracy', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[1], data=df_best[(df_best['weights'] == 'random') & (df_best['pooling'] == 'max')], x='image_width', y='simple_f1', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[2], data=df_best[(df_best['weights'] == 'random') & (df_best['pooling'] == 'max')], x='image_width', y='simple_precision', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[3], data=df_best[(df_best['weights'] == 'random') & (df_best['pooling'] == 'max')], x='image_width', y='simple_recall', hue='weights', style='pooling', markers=True)
for ax in axes:
    ax.set(xscale="log")
    ax.set(xticks=df_best['image_width'].unique())
    ax.set(xticklabels=df_best['image_width'].unique())
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Invasive vs. none-invasive')
axes = axes.flatten()
sns.lineplot(ax=axes[0], data=df_best, x='image_width', y='simple_accuracy', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[1], data=df_best, x='image_width', y='simple_f1', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[2], data=df_best, x='image_width', y='simple_precision', hue='weights', style='pooling', markers=True)
sns.lineplot(ax=axes[3], data=df_best, x='image_width', y='simple_recall', hue='weights', style='pooling', markers=True)
for ax in axes:
    ax.set(xscale="log")
    ax.set(xticks=df_best['image_width'].unique())
    ax.set(xticklabels=df_best['image_width'].unique())
plt.show()


df_last = df_all.groupby(['weights','pooling','image_width'])['val_accuracy'].last().reset_index()
ax = sns.lineplot(data=df_last, x='image_width', y='val_accuracy', hue='weights', style='pooling', markers=True)
ax.set(xscale="log")
ax.set(xticks=df_last['image_width'].unique())
ax.set(xticklabels=df_last['image_width'].unique())
plt.show()

print(df_all)