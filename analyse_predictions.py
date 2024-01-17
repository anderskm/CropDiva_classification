import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.special import softmax
from scipy.stats import norm
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import utils

network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_NormCW_less_dataAug__0e244979'
network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_RetrainExtraDataItr1__98add71d'
network_folder_path = 'networks_important/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_16OCT_Approved__527e4206' # th100
network_folder_path = 'networks_important/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_16OCT_Approved__e75fc964' # th65
dataset_folder = 'Datasets'
subdataset = 'validation'

network_name = os.path.split(network_folder_path)[-1]
unique_identifier = network_name.split('__')[-1]

# Load configuration file from trained network
with open(os.path.join(network_folder_path, 'configuration.json'),'r') as fob:
    train_config = json.load(fob)

print('\nTraining configuration: ', train_config)

# Load labels dict
labels_dict_file = os.path.join(dataset_folder,train_config['dataset_id'],'labels_dict_' + train_config['dataset_id'] + '.json')
with open(labels_dict_file,'r') as fob:
    labels_dict = json.load(fob)
N_classes = len(labels_dict)
print('Classes: ', labels_dict)

# Read dataset
dataframe_filepath = glob.glob(os.path.join(dataset_folder, train_config['dataset_id'],'*_'+ train_config['dataset_id'] + '_' + subdataset.title() + '.pkl'))[0]
df = pd.read_pickle(dataframe_filepath)

# Load dataframe
df_filename = os.path.split(dataframe_filepath)[-1]
df_filename_parts = df_filename.split(sep='.')
df_filename_predictions = '.'.join(df_filename_parts[:-1]) + '_w_pred_'  + unique_identifier + '.pkl'
df = pd.read_pickle(os.path.join(network_folder_path, df_filename_predictions))

## Precision recall curves

fig, axs = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
axs = axs.flatten()
fig2, axs2 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False)
axs2 = axs2.flatten()
df_ths_list = []
all_pred_all_labels = np.concatenate(df['predictions'].values)
# all_pred_all_labels = softmax(all_pred_all_labels, axis=1)
distribution_param_dict = {}
EPPOs = df['EPPO'].unique()
EPPOs.sort()
for eppo, ax, ax2 in zip(EPPOs, axs, axs2):
    label_idx = labels_dict[eppo]
    df_eppo, _ = utils.dataframe_filtering(df, df['EPPO'] == eppo)
    eppo_pred = np.concatenate(df_eppo['predictions'].values)[:,label_idx]
    all_pred = all_pred_all_labels[:,label_idx]
    # print(eppo, df_eppo.shape)
    precision, recall, threshold = precision_recall_curve(df['label_no']==label_idx, all_pred)
    th_idx = len(precision) - next(i for i,x in enumerate(np.flip(precision)) if x<1)
    ax.plot(recall, precision)
    if th_idx >= len(precision)-1:
        th_idx = -1
    # print(eppo, threshold[th_idx], precision[th_idx], recall[th_idx])
    # print(threshold[th_idx:-1])
    # print(np.median(threshold[th_idx:-1]))
    th_med = np.median(threshold[th_idx:-1])
    th_optim = threshold[th_idx]
    print(eppo, '\t', len(eppo_pred), '\tmedian', '\t',  th_med, '\t',  norm.cdf(th_med, loc=np.mean(eppo_pred), scale=np.std(eppo_pred)))
    print(eppo, '\t',  len(eppo_pred), '\toptim', '\t',  th_optim, '\t',  norm.cdf(th_optim, loc=np.mean(eppo_pred), scale=np.std(eppo_pred)))
    ax.plot(recall[th_idx], precision[th_idx],'x', label='th=' + str(threshold[th_idx]))
    ax.set_title(eppo + '(N=' + str(df_eppo.shape[0]) + ')')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left')

    # Plot histogram
    n, bins, patches = ax2.hist(eppo_pred, bins=30, density=True)
    mu = np.mean(eppo_pred)
    sigma = np.std(eppo_pred)
    distribution_param_dict[eppo] = {'mu': mu, 'sigma': sigma}
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax2.plot(bins, y, '--')
    ax2.plot(th_med*np.ones((2,)), [0, 0.15],'-', label='th_median ({0:.2f})'.format(norm.cdf(th_med, loc=np.mean(eppo_pred), scale=np.std(eppo_pred))))
    ax2.plot(th_optim*np.ones((2,)), [0, 0.15],'-', label='th_optim ({0:.2f})'.format(norm.cdf(th_optim, loc=np.mean(eppo_pred), scale=np.std(eppo_pred))))
    ax2.set_title(eppo + '(N=' + str(df_eppo.shape[0]) + ')')
    ax2.legend(loc='upper right')

        
    df_ths_list.append(pd.DataFrame.from_dict({'EPPO': [eppo],
                                               'N_samples': df_eppo.shape[0],
                                                'Threshold_optim': th_optim, #[threshold[th_idx]],
                                                'Threshold_median': th_med,
                                                'N_optim': len(threshold[th_idx:-1]),
                                                'Precision': [precision[th_idx]],
                                                'Recall': [recall[th_idx]],
                                                'mu': mu,
                                                'sigma': sigma
                                            }))
df_ths = pd.concat(df_ths_list)
df_ths = df_ths.set_index(pd.Index([labels_dict[eppo] for eppo in df_ths['EPPO'].values])).sort_index()

    # display = PrecisionRecallDisplay.from_predictions(df['label_no']==label_idx, all_pred, ax=ax, name=eppo + '(N=' + str(df_eppo.shape[0]) + ')')
print(df_ths)
plt.show()

df_filename_opt_thresholds = '.'.join(df_filename_parts[:-1]) + '_network_'  + unique_identifier + '_optim_thresholds.pkl'
df_ths.to_pickle(os.path.join(network_folder_path, df_filename_opt_thresholds))
df_ths.to_json(os.path.join(network_folder_path, df_filename_opt_thresholds + '.json'), orient='index', indent=3)


## Apply on test set
# Read dataset
dataframe_filepath = glob.glob(os.path.join(dataset_folder, train_config['dataset_id'],'*_'+ train_config['dataset_id'] + '_test.pkl'))[0]
df = pd.read_pickle(dataframe_filepath)

# Load dataframe
df_filename = os.path.split(dataframe_filepath)[-1]
df_filename_parts = df_filename.split(sep='.')
df_filename_predictions = '.'.join(df_filename_parts[:-1]) + '_w_pred_' + unique_identifier + '.pkl'
df_test = pd.read_pickle(os.path.join(network_folder_path, df_filename_predictions))

print('Evaluation dataset size:', df_test.shape[0])

# Classify samples
eval_pred_values = np.concatenate(df_test['predictions'].values)
eval_pred_max_flag = np.equal(eval_pred_values, np.expand_dims(np.max(eval_pred_values, axis=1), axis=1))
eval_pred_above_thresh_flag = eval_pred_values > df_ths['Threshold_median'].values
cl = np.where(np.logical_and(eval_pred_above_thresh_flag, eval_pred_max_flag))
# cl[0] --> row index in dataframe, where prediction is greater than threshold
# cl[1] --> class index in corresponding row in dataframe, where prediction is greater than threshold
print('In liners: ', len(cl[0]))

# Find misclassifications
miss_class = np.where(df_test['label_no'].values[cl[0]] != cl[1])

# Dataframe with misclassifications
df_test_miss = df_test.iloc[cl[0][miss_class[0]].tolist(),]
idx_to_label_dict = {v: k for k, v in labels_dict.items()}
df_test_miss.loc[:,'pred_label'] = [idx_to_label_dict[i] for i in df_test_miss['pred_label_no'].values] 
print('Misclassification', df_test_miss.shape[0])
print(df_test_miss)
print(df_test_miss[['label','pred_label']])
print(labels_dict)

print('done')
