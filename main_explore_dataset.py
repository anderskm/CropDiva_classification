# import argparse
# from datetime import datetime
# import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import scipy.spatial.distance
from scipy.stats import chisquare
import seaborn as sns
import tqdm

import utils

input_dataframe_filename = 'dataframe_annotations__filtered.pkl'

df = pd.read_pickle(input_dataframe_filename)

# def print_annotation_stats(df):
#     print('\nNumber of images with each label:')
#     df_img_per_class = df.groupby(['label','Date'])['label'].count().unstack().fillna(0).astype(np.uint16)
#     df_img_per_class['Sum'] = df_img_per_class.sum(axis=1)
#     df_img_per_class = df_img_per_class.append(df_img_per_class.sum(axis=0).rename('Sum'))
#     print(df_img_per_class)

#     print('\nNumber of polygons with each label:')
#     df_poly_per_class = df.groupby(['label','Date'])['N_polygons'].sum().unstack().fillna(0).astype(np.uint16)
#     df_poly_per_class['Sum'] = df_poly_per_class.sum(axis=1)
#     df_poly_per_class = df_poly_per_class.append(df_poly_per_class.sum(axis=0).rename('Sum'))
#     print(df_poly_per_class)

#     print('\nNumber of clusters:')
#     print(df['cluster'].nunique())

#     print('\nNumber of clusters each label appear in:')
#     print(df.groupby(['label'])['cluster'].nunique())

#     print('\nNumber of images per cluster each label appear in:')
#     print(df.groupby(['label'])['label'].count()/df.groupby(['label'])['cluster'].nunique())

#     # TODO: Polygon sizes/area per species

# def dataframe_filtering(df, mask):
#     df_keep = df[mask]
#     df_removed = df[~mask]
#     return df_keep, df_removed

utils.print_annotation_stats(df)

df['Polygon_area_rel'] = df['Polygon_area'] / (3036*4048)

df_filt_wo_NoneMixed = df[(df['label'] != 'None') & (df['label'] != 'Mixed')]
df_filt_wo_None = df[df['label'] != 'None']


sns.histplot(df_filt_wo_NoneMixed, x='Polygon_area_rel', hue='label', stat='probability', log_scale=False, element='poly', fill=False, common_norm=False, kde=False, cumulative=True)
plt.show()

g = sns.FacetGrid(df_filt_wo_None,col='label', col_wrap=3)
g.map(sns.histplot, 'Polygon_area_rel', stat='probability', log_scale=False, element='bars', fill=False, common_norm=False, kde=True, bins=20, binrange=(0.0, 1.0))
plt.show()


fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='label', animation_frame='dates', mapbox_style='carto-positron')
fig1.show()

fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='dates', animation_frame='label', mapbox_style='carto-positron')
fig1.show()

# fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='cluster', mapbox_style='carto-positron', animation_frame='label', color_discrete_sequence=px.colors.qualitative.Alphabet)
# fig1.show()


## TODO: Average polygon for each species: https://gis.stackexchange.com/a/68617

print('done')