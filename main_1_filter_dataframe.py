import glob
import numpy as np
import os
import pandas as pd

import utils

input_dataframe_filename = 'dataframe_annotations__all.pkl'
output_dataframe_filename = 'dataframe_annotations__filtered.pkl'
img_folder = 'D:/VJD_data/resize_0192x0256' #'C:/Vejdirektorat/dataset_exploration/images_for_annotation'

df_all = pd.read_pickle(input_dataframe_filename)

print('\n\n### Dataframe stats BEFORE filtering ###')
utils.print_annotation_stats(df_all)

## Clean-up

print('\n\n### Cleaning up dataset ###')

# Remove images from classes with few samples
print('\nRemoving classes with too few samples')
df_label_count = df_all.groupby(['label'])['image'].count()
df_filt = df_all
labels = df_label_count.index.to_list()
label_count = df_label_count.to_list()
labels_discard = [l for l,c in zip(labels, label_count) if c < 100]

for label in labels_discard:
    df_filt, _ = utils.dataframe_filtering(df_filt, df_filt['label'] != label)


# for label, count in zip(labels, label_count):
#     if (count <= 100):
#         df_filt, _ = utils.dataframe_filtering(df_filt, 
#                                                 df_filt['label'] == label)



# Filter images with underrepresented labels and images with mixed labels
# print('\nRemoving images with labels Mixed and Bjoerneklo:')
# df_filt_lbl, df_removed_lbl = utils.dataframe_filtering(df_all,
#                                                         (df_all['label'] != 'Mixed') & (df_all['label'] != 'Bjoerneklo'))
# print(df_removed_lbl.groupby(['label','dates'])['label'].count().unstack())

# # Remove images with polygons (bounding box) large overlap
# # (Note: In some cases, the polygons are not overlapping, but their BB are.)
# print('\nRemoving images bounding box overlap > 75%:')
# df_filt_bb, df_removed_bb = utils.dataframe_filtering(df_filt_lbl, 
#                                                         df_filt_lbl['BBox_IoA_max'] <= 0.75)
# print(df_removed_bb.groupby(['label','dates'])['label'].count().unstack())

# # Filter on image names
# if img_folder:
#     subfolders = glob.glob(os.path.join(img_folder,'*'))
#     image_names = []
#     for subfolder in subfolders:
#         image_paths = glob.glob(os.path.join(subfolder, '*.jpg'))
#         image_names += [os.path.split(image_path)[1] for image_path in image_paths]
#     df_filt = df_filt_bb[df_filt_bb['image'].isin(image_names)]
# else:
#     df_filt = df_filt_bb

# For small dataset. Assign all images to their own individual cluster
# df_filt['cluster'] = [i for i in range(len(df_filt))]

# # Show polygons with overlapping BB
# df_ol = df_filt[df_filt['BBox_IoA_max'] > 0.75]
# for r, row in df_ol.iterrows():
#     polygons = row['polygons']
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot([0, 4048, 4048, 0, 0], [0,0, 3036, 3036, 0])
#     print(ax)
#     for p, polygon in enumerate(polygons):
#         # TODO: Plot polygon
#         ax.plot(polygon[:,0], polygon[:,1])
#         ax.text(polygon[:,0].mean(), polygon[:,1].mean(), row['labels'][p])
#     ax.axis('equal')
#     ax.set_title(str(row['BBox_IoA_max']))
#     plt.show()

print('\n### End of Cleaning up dataset ###\n\n')

## End of Clean-up

print('\n\n### Dataframe stats AFTER filtering ###')
utils.print_annotation_stats(df_filt)

df_filt.to_pickle(output_dataframe_filename)

print('done')