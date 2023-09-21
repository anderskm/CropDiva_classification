import argparse
import glob
import json
import numpy as np
import os
import pandas as pd

import utils

def main():
    # Setup input argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_df', action='store', default='dataframe_annotations__all.pkl', type=str, help='Filename of pickle file containing dataframe to be filtered (default: %(default)s).')
    parser.add_argument('--output_df', action='store', default='', type=str, help='Filename of filtered dataframe. Leave empty to simply append "__filtered" to the filename (default: %(default)s).')
    parser.add_argument('--discard_labels_file', action='store', default='', type=str, help='Filename of json file with list of classes to discard. Leave empty to remove classes with less than 100 samples (default: %(default)s).')

    args = vars(parser.parse_known_args()[0])
    print('\nArguments: ', args)

    input_dataframe_filename = args['input_df']
    print('Input: ', input_dataframe_filename)
    output_dataframe_filename = args['output_df']
    basename, ext = os.path.splitext(os.path.basename(input_dataframe_filename))
    if not output_dataframe_filename:
        output_dataframe_filename = os.path.join(os.path.dirname(input_dataframe_filename), basename + '__filtered' + ext)
    print('Output: ', output_dataframe_filename)
    labels_discard_file = args['discard_labels_file']
    if not labels_discard_file:
        labels_discard_file = os.path.join(os.path.dirname(input_dataframe_filename), basename + '__labels_discard.json')
    
    df_all = pd.read_pickle(input_dataframe_filename)

    print('\n\n### Dataframe stats BEFORE filtering ###')
    utils.print_annotation_stats(df_all)

    ## Clean-up

    print('\n\n### Cleaning up dataset ###')

    # Remove images from classes with few samples
    print('\nRemoving classes with too few samples')

    if os.path.exists(labels_discard_file):
        with open(labels_discard_file,'r') as fob:
            labels_discard = json.load(fob)
    else:
        df_label_count = df_all.groupby(['label'])['image'].count()
        
        labels = df_label_count.index.to_list()
        label_count = df_label_count.to_list()
        labels_discard = [l for l,c in zip(labels, label_count) if c < 100]
        with open(labels_discard_file,'w') as fob:
            json.dump(labels_discard, fob)
    
    df_filt = df_all
    for label in labels_discard:
        df_filt, _ = utils.dataframe_filtering(df_filt, df_filt['label'] != label)

    print('\n### End of Cleaning up dataset ###\n\n')

    ## End of Clean-up

    print('\n\n### Dataframe stats AFTER filtering ###')
    utils.print_annotation_stats(df_filt)

    df_filt.to_pickle(output_dataframe_filename)

    print('done')

if __name__ == '__main__':
    main()