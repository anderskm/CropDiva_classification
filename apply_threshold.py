import confusionmatrix as CM
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from scipy.stats import norm

import utils

# network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_NormCW_less_dataAug__0e244979'
network_folder_path = 'networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_RetrainExtraDataItr1__98add71d'
network_folder_path = 'networks_important/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_16OCT_Approved__527e4206' # th100
network_folder_path = 'networks_important/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_16OCT_Approved__e75fc964' # th65
dataset_folder = 'Datasets'
# predictions_file = 'dataframe_annotations_87bb031e36e9d24636e9d246_Train_w_pred_.pkl'
predictions_file = 'dataframe_annotations_87bb031e36e9d24636e9d246_Train_w_pred_.pkl'
predictions_file = 'dataframe_annotations_94cd2a4536e9d24636e9d246_Train_w_pred_.pkl'
predictions_file = 'dataframe_annotations_42c2408d36e9d24636e9d246_Train_w_pred_.pkl' # th65, UnApproved
# predictions_file = 'dataframe_annotations_7ba57315338fd2052c3d8b07_Train_w_pred_e75fc964.pkl' #th65, Train
# predictions_file = 'dataframe_annotations_7ba57315338fd2052c3d8b07_Test_w_pred_e75fc964.pkl' #th65, Test
# predictions_file = 'dataframe_annotations_7ba57315338fd2052c3d8b07_Validation_w_pred_e75fc964.pkl' #th65, Validation

negate_misclassification_confidence = False

test_labels_dict_file = 'Datasets/labels_dict_15.json' # If additional labels are used in the evaluated dataset. Leave blank if same.

## Do not change below

network_id = network_folder_path.split('_')[-1]

# Load configuration file from trained network
with open(os.path.join(network_folder_path, 'configuration.json'),'r') as fob:
    train_config = json.load(fob)

print('\nTraining configuration: ', train_config)

# Load labels dict
labels_dict_file = os.path.join(dataset_folder, train_config['dataset_id'],'labels_dict_' + train_config['dataset_id'] + '.json')
with open(labels_dict_file,'r') as fob:
    labels_dict = json.load(fob)
N_classes = len(labels_dict)
label_no_2_label = dict([(value, key) for key, value in labels_dict.items()])
print('Classes: ', labels_dict)

if test_labels_dict_file:
    with open(test_labels_dict_file,'r') as fob:
        labels_dict_test = json.load(fob)
else:
    labels_dict_test = labels_dict

th_file = glob.glob(os.path.join(network_folder_path, '*_thresholds.pkl'))[0]
df_ths = pd.read_pickle(th_file)

df_predictions = pd.read_pickle(os.path.join(network_folder_path, predictions_file))

df_predictions['pred_label'] = [label_no_2_label[lo] for lo in df_predictions['pred_label_no'].values.tolist()]

# Logits to softmax
df_predictions['softmaxes'] = df_predictions[['predictions']].apply(lambda x: softmax(x[0]).squeeze(), axis=1)
df_predictions['softmax'] = np.concatenate(df_predictions['softmaxes'].values).reshape((-1,N_classes))[np.arange(df_predictions.shape[0]), df_predictions['pred_label_no']]

# Logits to probabilities according to class distributions
eval_pred_values = np.concatenate(df_predictions['predictions'].values)
Ps = []
for idx, row in df_ths.iterrows():
    Ps.append(norm.cdf(eval_pred_values[:, idx], loc=row['mu'], scale=row['sigma']).reshape(-1,1))
df_predictions['CDFs'] = np.concatenate(Ps, axis=1).tolist()
df_predictions['CDF'] = np.concatenate(df_predictions['CDFs'].values).reshape((-1,N_classes))[np.arange(df_predictions.shape[0]), df_predictions['pred_label_no']]

df_predictions['confidence'] = df_predictions['softmax']*df_predictions['CDF']

if negate_misclassification_confidence:
    df_predictions['confidence'] = ((-1*(df_predictions['label_no'] != df_predictions['pred_label_no']) + (df_predictions['label_no'] == df_predictions['pred_label_no']))*df_predictions['confidence'])
    print('# misclassifications: ', (df_predictions['label_no'] != df_predictions['pred_label_no']).sum())

labels = [key for key in labels_dict_test.keys()]
CM_pred = CM.confusionmatrix(len(labels), labels)
CM_pred.Append(df_predictions['label_no'], df_predictions['pred_label_no'])
print('\nConfusion matrix from all samples:')
print(CM_pred)

# Classify samples
eval_pred_values = np.concatenate(df_predictions['predictions'].values)
eval_pred_max_flag = np.equal(eval_pred_values, np.expand_dims(np.max(eval_pred_values, axis=1), axis=1))
# eval_pred_above_thresh_flag = eval_pred_values > df_ths['Threshold_median'].values
eval_pred_above_thresh_flag = eval_pred_values > df_ths['Threshold_optim'].values
cl = np.where(np.logical_and(eval_pred_above_thresh_flag, eval_pred_max_flag))
# cl[0] --> row index in dataframe, where prediction is greater than threshold
# cl[1] --> class index in corresponding row in dataframe, where prediction is greater than threshold
print('In liners: ', len(cl[0]))

pred_id = predictions_file.split('_')[2]

df_inliers = df_predictions.iloc[cl[0],]

# fig, axs = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False)
# axs = axs.flatten()
# fig2, axs2 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False)
# axs2 = axs2.flatten()
# fig3, axs3 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=False)
# axs3 = axs3.flatten()
# fig4, axs4 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
# axs4 = axs4.flatten()
# fig5, axs5 = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
# axs5 = axs5.flatten()
# for idx, row in df_ths.iterrows():
#     ax = axs[idx]
#     ax2 = axs2[idx]
#     ax3 = axs3[idx]
#     ax4 = axs4[idx]
#     ax5 = axs5[idx]
#     df_eppo_pred,_ = utils.dataframe_filtering(df_predictions, df_predictions['pred_label'] == row['EPPO'])
#     df_eppo_inli,_ = utils.dataframe_filtering(df_inliers, df_inliers['pred_label'] == row['EPPO'])
#     if not df_eppo_pred.empty:
#         n, bins, patches = ax.hist(df_eppo_pred['softmax'], bins=100, density=False)
#         n, bins2, patches = ax2.hist(df_eppo_pred['CDF'], bins=100, density=False)
#         n, bins3, patches = ax3.hist(df_eppo_pred['confidence'], bins=100, density=False)
#         ax4.plot(df_eppo_pred['CDF'], df_eppo_pred['confidence'],'.')
#         ax5.plot(df_eppo_pred['softmax'], df_eppo_pred['confidence'],'.')
#         ax.set_title(row['EPPO'] + '(N=' + str(df_eppo_inli.shape[0]) + '/' + str(df_eppo_pred.shape[0]) + ')')
#         ax2.set_title(row['EPPO'] + '(N=' + str(df_eppo_inli.shape[0]) + '/' + str(df_eppo_pred.shape[0]) + ')')
#         ax3.set_title(row['EPPO'] + '(N=' + str(df_eppo_inli.shape[0]) + '/' + str(df_eppo_pred.shape[0]) + ')')
#         ax4.set_title(row['EPPO'] + '(N=' + str(df_eppo_inli.shape[0]) + '/' + str(df_eppo_pred.shape[0]) + ')')
#         ax5.set_title(row['EPPO'] + '(N=' + str(df_eppo_inli.shape[0]) + '/' + str(df_eppo_pred.shape[0]) + ')')
#         if not df_eppo_inli.empty:
#             n, bins, patches = ax.hist(df_eppo_inli['softmax'], bins=bins, density=False)
#             n, bins, patches = ax2.hist(df_eppo_inli['CDF'], bins=bins2, density=False)
#             n, bins, patches = ax3.hist(df_eppo_inli['confidence'], bins=bins3, density=False)
#             ax4.plot(df_eppo_inli['CDF'], df_eppo_inli['confidence'],'.')
#             ax5.plot(df_eppo_inli['softmax'], df_eppo_inli['confidence'],'.')
#         ax.set_xlabel('softmax')
#         ax.set_ylabel('Count')
#         ax2.set_xlabel('CDF')
#         ax2.set_ylabel('Count')
#         ax3.set_xlabel('Confidence')
#         ax3.set_ylabel('Count')
#         ax4.set_xlabel('CDF')
#         ax4.set_ylabel('Confidence')
#         ax5.set_xlabel('Softmax')
#         ax5.set_ylabel('Confidence')
# plt.show()

g = sns.FacetGrid(df_predictions, col="label", col_wrap=4)
g.map(sns.histplot, "confidence", stat='probability', binwidth=0.05, binrange=[0.0, 1.0])

g = sns.FacetGrid(df_predictions, col="pred_label", col_wrap=4)
g.map(sns.histplot, "confidence", stat='probability', binwidth=0.05, binrange=[0.0, 1.0])

inliners_path = os.path.join(network_folder_path, 'inliers__net_' + network_id + '_train_' + train_config['dataset_id'] + '_pred_' + pred_id + '.pkl')
df_inliers.to_pickle(inliners_path)

labels = [key for key in labels_dict_test.keys()]
CM_inliers = CM.confusionmatrix(len(labels), labels)
CM_inliers.Append(df_inliers['label_no'], df_inliers['pred_label_no'])
print('\nConfusion matrix from inliers:')
print(CM_inliers)



# Find misclassifications
miss_class = np.where(df_predictions['label_no'].values[cl[0]] != cl[1])

# Dataframe with misclassifications
df_predictions_miss = df_predictions.iloc[cl[0][miss_class[0]].tolist(),]
idx_to_label_dict = {v: k for k, v in labels_dict.items()}
df_predictions_miss.loc[:,'pred_label'] = [idx_to_label_dict[i] for i in df_predictions_miss['pred_label_no'].values] 
print('Misclassification', df_predictions_miss.shape[0])
print(df_predictions_miss)
print(df_predictions_miss[['label','pred_label']])

columns_to_export = ['image','UploadID','ImageID','BBoxID','label','pred_label','softmax','CDF','confidence']

predictions_out_path = os.path.join(network_folder_path, 'ALL__net_' + network_id + '_train_' + train_config['dataset_id'] + '_pred_' + pred_id + '.csv')
df_predictions.to_csv(path_or_buf=predictions_out_path, columns=columns_to_export, header=True, index=False, mode='x', encoding='utf-8', compression=None)

inliers_out_path = os.path.join(network_folder_path, 'INLIERS__net_' + network_id + '_train_' + train_config['dataset_id'] + '_pred_' + pred_id + '.csv')
df_inliers.to_csv(path_or_buf=inliers_out_path, columns=columns_to_export, header=True, index=False, mode='x', encoding='utf-8', compression=None)


print('done')
