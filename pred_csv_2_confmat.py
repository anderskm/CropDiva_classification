import confusionmatrix as cm
import json
import numpy as np
import pandas as pd


csv_file = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/CropDiva_classification/networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_RetrainExtraDataItr1__98add71d/ALL__net_98add71d_train_1c15f53cfb2ae5729e2c5b67_pred_87bb031e36e9d24636e9d246.csv'

labels_dict_file = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/CropDiva_classification/Datasets/4a64dfabd48e2cf59ba9cc59/labels_dict_4a64dfabd48e2cf59ba9cc59.json'

approved_labels_df = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/CropDiva_classification/dataframe_annotations__filtered.pkl'

with open(labels_dict_file,'r') as fob:
    labels_dict = json.load(fob)

df = pd.read_csv(csv_file)

df['label_no'] = [labels_dict[lo] for lo in df['label'].values.tolist()]
df['pred_label_no'] = [labels_dict[lo] for lo in df['pred_label'].values.tolist()]

cmat = cm.confusionmatrix(len(labels_dict), labels=[k for k in labels_dict])
cmat.Append(df['label_no'], df['pred_label_no'])
print(cmat)

df_approved = pd.read_pickle(approved_labels_df)
df_approved['label_no'] = [labels_dict[lo] for lo in df_approved['label'].values.tolist()]

cmat_approved = cm.confusionmatrix(len(labels_dict), labels=[k for k in labels_dict])
cmat_approved.Append(df_approved['label_no'], df_approved['label_no'])
print(cmat_approved)

df_label_count = pd.DataFrame({'label': [k for k in labels_dict], 'existing': cmat_approved.predictedCounts(), 'potential': cmat.predictedCounts() })

print(df_label_count)

print('done')
