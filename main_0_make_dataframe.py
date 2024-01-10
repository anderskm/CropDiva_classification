import argparse
import glob
import os
import pandas as pd
from PIL import Image
import tqdm

import utils

def get_subfolders(main_folder):
    main_folder_content = os.listdir (main_folder)
    subfolders = []
    for obj in main_folder_content:
        obj_path = os.path.join(main_folder, obj)
        if os.path.isdir(obj_path):
            subfolders.append(obj)
    return subfolders

def main():
    # Setup input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_df', action='store', default='dataframe_annotations__all.pkl', type=str, help='Filename of pickle file containing dataframe. (default: %(default)s).')
    parser.add_argument('--data_folder', action='store', default='', type=str, help='Path to folder containing the image data, where each subfolder corresponds to a specific class (default: %(default)s).')
    parser.add_argument('--img_ext', action='store', default='png', type=str, help='File extention of images. (default: %(default)s).')

    # Parse inputs
    args = vars(parser.parse_known_args()[0])
    print('\nArguments: ', args)
    data_folder = args['data_folder']
    print('Input data folder: ', data_folder)
    img_ext = args['img_ext']
    print('Image extension: ', img_ext)
    output_dataframe_filename = args['output_df']
    print('Output file: ', output_dataframe_filename)

    # Get all subfolders in main data folder
    image_folders = get_subfolders(data_folder)

    # Loop through classes/subfolders
    DFs = []
    for img_folder in tqdm.tqdm(image_folders, desc='Parsing subfolders'):
        # Find all images in subfolder
        image_list = glob.glob(os.path.join(data_folder, img_folder,'*.' + img_ext))

        # Initialize dataframe for class subfolder
        df_bounding_boxes_list = []
        df_bounding_boxes_list.append(pd.DataFrame(columns=['image','folder','EPPO','UploadID','ImageID','BBoxID','width','height','area','label']))
        for image_path in tqdm.tqdm(image_list, desc='Parsing ' + img_folder, leave=False):
            folder, image_name = os.path.split(image_path)
            
            # Extract meta data from filename
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
            # Store image meta data as a row in a data frame
            df_bounding_boxes_list.append(pd.DataFrame.from_dict({'image': [image_name],
                                                                    'folder': [img_folder],
                                                                    'EPPO': [EPPO],
                                                                    'UploadID': [UploadID],
                                                                    'ImageID': [ImageID],
                                                                    'BBoxID': [BBoxID],
                                                                    'width': [width],
                                                                    'height': [height],
                                                                    'area': [area],
                                                                    'label': [label]
                                                                }))
        # Concatenate the individual image dataframes into a single dataframe for the class
        df_bounding_boxes = pd.concat(df_bounding_boxes_list)
        DFs.append(df_bounding_boxes)
    
    # Concatenate the dataframes from each subfolder into a single dataframe containing all classes
    df_all = pd.concat(DFs, ignore_index=True)
    
    # Save as pickle file
    df_all.to_pickle(output_dataframe_filename)
    print('Dataframe saved to: ', output_dataframe_filename)

    # Print a summary
    utils.print_annotation_stats(df_all)

    print('done')

if __name__ == '__main__':
    main()
