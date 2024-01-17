import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import pandas as pd

pred_A_file = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/CropDiva_classification/networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_NormCW_less_dataAug__0e244979/ALL__net_0e244979_train_4a64dfabd48e2cf59ba9cc59_pred_87bb031e36e9d24636e9d246.csv'
pred_B_file = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/CropDiva_classification/networks/EfficientNetV2S_imagenet_224x224_avg_adam_lr0.0001_bs32_RetrainExtraDataItr1__98add71d/ALL__net_98add71d_train_1c15f53cfb2ae5729e2c5b67_pred_87bb031e36e9d24636e9d246.csv'

df_A = pd.read_csv(pred_A_file)
df_A = df_A.set_index('image')

df_B = pd.read_csv(pred_B_file)
df_B = df_B.set_index('image')

df_merged = pd.merge(df_A, df_B, how='outer', on='image', suffixes=('_A','_B'))

gamma = 0.3
fig, axs = plt.subplots(nrows=1, ncols=3)
# axs[0].plot(df_merged['softmax_A'],df_merged['softmax_B'],'.')
# axs[1].plot(df_merged['CDF_A'],df_merged['CDF_B'],'.')
# axs[2].plot(df_merged['confidence_A'],df_merged['confidence_B'],'.')
axs[0].hist2d(df_merged['softmax_A'],df_merged['softmax_B'], norm=mcolors.PowerNorm(gamma))
axs[1].hist2d(df_merged['CDF_A'],df_merged['CDF_B'], norm=mcolors.PowerNorm(gamma))
axs[2].hist2d(df_merged['confidence_A'],df_merged['confidence_B'], norm=mcolors.PowerNorm(gamma))

fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True)
# axs[0].plot(df_merged['softmax_A'],df_merged['softmax_B'],'.')
# axs[1].plot(df_merged['CDF_A'],df_merged['CDF_B'],'.')
# axs[2].plot(df_merged['confidence_A'],df_merged['confidence_B'],'.')
axs[0].hist(df_merged['softmax_A'],bins=10, histtype='step', density=False, label='Approved')
axs[0].hist(df_merged['softmax_B'],bins=10, histtype='step', density=False, label='+Unapproved')
axs[1].hist(df_merged['CDF_A'],bins=10, histtype='step', density=False, label='Approved')
axs[1].hist(df_merged['CDF_B'],bins=10, histtype='step', density=False, label='+Unapproved')
axs[2].hist(df_merged['confidence_A'],bins=10, histtype='step', density=False, label='Approved')
axs[2].hist(df_merged['confidence_B'],bins=10, histtype='step', density=False, label='+Unapproved')
axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')
axs[2].legend(loc='upper left')
axs[0].set_xlabel('Softmax')
axs[1].set_xlabel('CDF')
axs[2].set_xlabel('Confidence')
axs[0].set_ylabel('Count')

plt.show()

print('done')
