import glob
# import numpy as np
import os
import pandas as pd
from PIL import Image
import tqdm
# import warnings

import utils

# warnings.simplefilter('always')

output_dataframe_filename = 'dataframe_annotations__all.pkl'

data_folder = 'C:/Projekter/2023_CropDiva__Weed_classificaiton/TrainingData/05SEP23_raw'

def get_subfolders(main_folder):
    main_folder_content = os.listdir (main_folder)
    subfolders = []
    for obj in main_folder_content:
        obj_path = os.path.join(main_folder, obj)
        if os.path.isdir(obj_path):
            subfolders.append(obj)
    return subfolders

image_folders = get_subfolders(data_folder)

DFs = []

# Load annotations
for img_folder in tqdm.tqdm(image_folders, desc='Parsing subfolders'):
    image_list = glob.glob(os.path.join(data_folder, img_folder,'*.png'))

    df_bounding_boxes = pd.DataFrame(columns=['image','folder','EPPO','UploadID','ImageID','BBoxID','width','height','area','label'])
    for image_path in tqdm.tqdm(image_list, desc='Parsing ' + img_folder, leave=False):
        folder, image_name = os.path.split(image_path)

        image_name_parts = os.path.splitext(image_name)[0].split('_')
        EPPO = image_name_parts[0]
        UploadID = image_name_parts[1]
        ImageID = image_name_parts[2]
        BBoxID = image_name_parts[3]

        # Load image to get size
        with Image.open(image_path) as im:
            width, height = im.size
            area = width*height

        label = img_folder

        df_bounding_boxes = df_bounding_boxes.append({'image': image_name, 
                                                      'folder': img_folder, 
                                                      'EPPO': EPPO, 
                                                      'UploadID': UploadID, 
                                                      'ImageID': ImageID, 
                                                      'BBoxID': BBoxID,
                                                      'width': width,
                                                      'height': height,
                                                      'area': area,
                                                      'label': label
                                                      }, ignore_index=True)
    
    DFs.append(df_bounding_boxes)

df_all = pd.concat(DFs, ignore_index=True)

df_all.to_pickle(output_dataframe_filename)

utils.print_annotation_stats(df_all)

print('done')