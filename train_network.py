import argparse
import ast
import datetime
import glob
import GPUtil
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage.io
import skimage.transform
# from skimage.io import imread
# from skimage.transform import resize
# import tensorflow as tf
import tqdm

import cnn_model
import confusionmatrix as CM

def main():
    
    # Setup input argument parser
    parser = argparse.ArgumentParser()

    parser_group_data = parser.add_argument_group('Data')
    parser_group_data.add_argument('--dataset_id', action='store', default='', type=str, help='ID of dataset used for training (default: %(default)s).')
    parser_group_data.add_argument('--dataset_folder', action='store', default='datasets', type=str, help='Path to main folder containing all datasets (patch must not including dataset ID) (default: %(default)s).')
    parser_group_data.add_argument('--image_folder', action='store', default='images', type=str, help='Path to main folder containing the images in subfolders (default: %(default)s).')
    parser_group_data.add_argument('--image_size', action='store', nargs=2)
    parser_group_data.add_argument('--stratify', action='store_true', help='Set flag to stratify the training data before training.')

    parser_group_base_network = parser.add_argument_group('Base network:')
    parser_group_base_network.add_argument('--basenet', action='store', default='ResNet50V2', type=str, help='Name of base network model to use for training (default: %(default)s).')
    parser_group_base_network.add_argument('--weights', action='store', default='random', type=str, help='Name of weights pretrained weights. "imagenet" or "random" for random initialization (default: %(default)s).')
    parser_group_base_network.add_argument('--pooling', action='store', default='avg', type=str, help='Type of pooling used at end of network (default: %(default)s).')
    parser_group_base_network.add_argument('--basenet_params', action='store', type=str, help='Additional parameters parsed to base network. Specified as a dict. See Keras documentation for available parameters for each base network.')

    parser_group_training = parser.add_argument_group('Training')
    parser_group_training.add_argument('--epochs', action='store', default=10, type=int, help='Number of training epochs to run before stopping (default: %(default)s).')
    parser_group_training.add_argument('--batch_size', action='store', default=32, type=int, help='Number of examples to include in each batch during training (default: %(default)s).')
    parser_group_training.add_argument('--finetune', action='store_true', help='Set flag to only fine-tune on the last classification layer.')
    parser_group_training.add_argument('--loss_func', action='store', default='CrossEntropy', type=str, help='Name of loss function to use during training.')
    parser_group_training.add_argument('--loss_params', action='store', type=str, help='Parameters parsed to loss function. Specified as a dict. See Keras documentation for available parameters.')
    parser_group_training.add_argument('--optimizer', action='store', default='Adam', type=str, help='Name of optimizer to use during training.')
    parser_group_training.add_argument('--optimizer_params', action='store', type=str, help='Parameters parsed to optimizer. Specified as a dict. See Keras documentation for available parameters.')

    parser_group_utils = parser.add_argument_group('Utilities')
    parser_group_utils.add_argument('--network_folder', action='store', default='networks', type=str, help="Main folder, where trained network should be saved (default: %(default)s)")
    parser_group_utils.add_argument('--save_model_before_training', action='store_true', help='Set flag to save model before training (Model will always be saved after training).')

    ## Parse input arguments
    args = vars(parser.parse_args())
    print(args)
    # Data
    dataset_ID = args['dataset_id']
    dataset_main_folder = args['dataset_folder']
    dataset_folder = os.path.join(dataset_main_folder, dataset_ID)
    image_folder = args['image_folder']
    image_size = [int(i) for i in args['image_size']]
    input_shape = tuple(image_size + [3])
    stratify_training_data = args['stratify']
    # Base network
    base_network_name = args['basenet']
    base_network_weights_str = args['weights']
    if base_network_weights_str.lower() == 'random':
        base_network_weights = None
    else:
        base_network_weights = base_network_weights_str
    base_network_pooling = args['pooling']
    if args['basenet_params'] is None:
        base_network_params_dict = {}
    else:
        base_network_params_dict = ast.literal_eval(args['basenet_params'])
    # Training
    batch_size = args['batch_size']
    epochs = args['epochs']
    fine_tune_only = args['finetune']
    loss_func_name = args['loss_func'].lower()
    if args['loss_params'] is None:
        loss_params_dict = {}
    else:
        loss_params_dict = ast.literal_eval(args['loss_params'])
    optimizer_name = args['optimizer'].lower()
    if args['optimizer_params'] is None:
        optimizer_params_dict = {}
    else:
        optimizer_params_dict = ast.literal_eval(args['optimizer_params'])
    # Utils
    network_folder_main = args['network_folder']
    save_model_before_training = args['save_model_before_training']


    # Convert args to network folder name
    network_folder_name_list = [base_network_name, base_network_weights_str.lower(), str(image_size[0]) + 'x' + str(image_size[1])]
    network_folder_name_list.append(base_network_pooling)
    if fine_tune_only:
        network_folder_name_list.append('finetune')
    network_folder_name_list.append(optimizer_name)
    if 'learning_rate' in optimizer_params_dict:
        network_folder_name_list.append('lr' + str(optimizer_params_dict['learning_rate']))
    network_folder_name_list.append('bs'+str(batch_size))
    hash_func = hashlib.blake2s(digest_size=4)
    hash_func.update(bytes(str(args) + str(datetime.datetime.now()), 'utf-8'))
    unique_identifier_from_args = hash_func.hexdigest()
    network_folder_name_list.append('_' + unique_identifier_from_args)
    network_folder_name = '_'.join(network_folder_name_list)
    print('Network folder name: ', network_folder_name)
    
    # Make network folder dir
    network_folder_path = os.path.join(network_folder_main, network_folder_name)
    os.makedirs(network_folder_path)
    # Dump configuration
    with open(os.path.join(network_folder_path, 'configuration.json'), 'w') as fob:
        json.dump(args, fob, indent=4)

    # Allocate a GPU
    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Get the first available GPU
    DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.01, maxMemory=0.01, attempts=999999, interval=60, verbose=True)
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    ## Setup datasets
    dataframe_training_file = glob.glob(os.path.join(dataset_folder,'*_'+ dataset_ID + '_Train.pkl'))[0] #'dataframe_annotations_5a8e1047f4d529e0607b50ba_Train.pkl'
    dataframe_validation_file = glob.glob(os.path.join(dataset_folder,'*_'+ dataset_ID + '_Validation.pkl'))[0]
    # dataframe_test_file = glob.glob(os.path.join(dataset_folder,'*_'+ dataset_ID + '_Test.pkl'))[0]

    df_train = pd.read_pickle(dataframe_training_file)
    df_validation = pd.read_pickle(dataframe_validation_file)
    # df_test = pd.read_pickle(dataframe_test_file)

    # Stratify training data
    df_image_label_count = df_train.groupby('label')['image'].count().to_frame()
    num_examples_per_label = np.asarray(df_image_label_count)
    if not 'alpha' in loss_params_dict:
        loss_params_dict['alpha'] = (1/num_examples_per_label).tolist() #np.max(np.log2(num_examples_per_label))-np.log2(num_examples_per_label)+1 #1/np.log2(num_examples_per_label)

    if stratify_training_data:
        scale_factors = np.round(np.max(num_examples_per_label)/num_examples_per_label).astype(int)
        df_image_label_count['scale_factor'] = scale_factors
        # Repeat the samples of each label by scale factor
        DFs = []
        for _, row in df_image_label_count.iterrows():
            DFs.append(pd.DataFrame(df_train[df_train['label'] == row.name].values.repeat(row['scale_factor'], axis=0), columns=df_train.keys()))
        # Merge dataframes for each label into new training dataframe
        df_train = pd.concat(DFs, ignore_index=True)
        df_train = df_train.sort_values(by=['label','UploadID','ImageID']) # Sort to get the images in somewhat same order as before
        print('Expected size of new dataframe:', np.prod(df_image_label_count.values,axis=1).sum())
        print('Actual size of new dataframe:', len(df_train))

    training_generator   = cnn_model.dataset_sample_generator(df_train, batch_size=batch_size, name='Training Generator', image_folder=image_folder, shuffle_end_of_epoch=True) #, image_size=input_shape)
    validation_generator = cnn_model.dataset_sample_generator(df_validation, batch_size=batch_size, name='Validation Generator', image_folder=image_folder, shuffle_end_of_epoch=False) #, image_size=input_shape)
    # test_generator       = cnn_model.dataset_sample_generator(df_test, batch_size=batch_size, name='Test Generator', image_folder=image_folder, shuffle_end_of_epoch=False) #, image_size=input_shape)

    # Load labels dict
    labels_dict_file = glob.glob(os.path.join(dataset_folder,'labels_dict_' + dataset_ID + '.json'))[0]
    with open(labels_dict_file,'r') as fob:
        labels_dict = json.load(fob)
    N_classes = len(labels_dict)
    print('Classes: ', labels_dict)
    print('Number of classes:', N_classes)

    print('\n')
    print('Theoretical accuracies (train, validation):')
    df_train_label_distribution = df_train.groupby('label')['image'].count()/len(df_train)
    df_validation_label_distribution = df_validation.groupby('label')['image'].count()/len(df_validation)
    # df_test_label_distribution = df_test.groupby('label')['image'].count()/len(df_test)
    train_most_frequent_class = df_train_label_distribution[df_train_label_distribution == df_train_label_distribution.max()].keys()[0]
    print('Uniform random guess: {train:.3f}, {validation:.3f}'.format(train=np.sum(np.multiply(1/float(N_classes), df_train_label_distribution)), validation=np.sum(np.multiply(1/float(N_classes), df_validation_label_distribution)) ))
    print('Always most frequent class ("' + train_most_frequent_class + '"): {train:.3f}, {validation:.3f}'.format(train=df_train_label_distribution[train_most_frequent_class], validation=df_validation_label_distribution[train_most_frequent_class]))
    print('Random guess w. train class distribution: {train:.3f}, {validation:.3f}'.format(train=np.sum(np.power(df_train_label_distribution,2)), validation=np.sum(np.multiply(df_train_label_distribution, df_validation_label_distribution))))
    print('\n')

    ## Setup network
    print('Building network...')
    model = cnn_model.build_model(input_shape=input_shape, N_classes=N_classes, basenet=base_network_name, weights=base_network_weights, pooling=base_network_pooling, fine_tune_only=fine_tune_only, **base_network_params_dict)
    # TODO: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
    model_callbacks = cnn_model.setup_callbacks(network_folder_path=network_folder_path)
    optimizer = cnn_model.setup_optimizer(optimizer_name, optimizer_params_dict)
    loss_func = cnn_model.setup_loss_func(loss_func_name, from_logits=True, loss_params_dict=loss_params_dict)
    print('Compiling network...')
    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy','categorical_accuracy'])

    print('Model input(s):')
    [print(i.shape, i.dtype) for i in model.inputs]
    print('Model output(s):')
    [print(o.shape, o.dtype) for o in model.outputs]
    print('Model layers:')
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]

    print('Input shape:', tuple(model.input.shape))
    print('Output shape:', tuple(model.output.shape))

    # Save model before fitting
    if save_model_before_training:
        print('Saving model before training...')
        model.save(os.path.join(network_folder_path, 'model_before_training'))

    ## Train network
    print('Fitting model...')
    # Setup callbacks
    # TODO: Move to cnn_model
    # csv_logger_callback = cnn_model.tf.keras.callbacks.CSVLogger(os.path.join(network_folder_path, 'training.log'))
    # checkpoint_callback = cnn_model.tf.keras.callbacks.ModelCheckpoint(os.path.join(network_folder_path, 'best_model_checkpoint'), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_freq='epoch')
    # tensorboard_callback = cnn_model.tf.keras.callbacks.TensorBoard(os.path.join(network_folder_path, 'logs'), histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=1)
    # model_callbacks = [csv_logger_callback, checkpoint_callback, tensorboard_callback]
    history = model.fit(training_generator, validation_data=validation_generator, epochs=epochs, verbose=1, callbacks=model_callbacks)
    print('Saving training history...')
    training_history = {'history': history.history, 'params': history.params, 'epoch': history.epoch}
    with open(os.path.join(network_folder_path, 'train_history.json'), 'w') as fob:
        json.dump(training_history, fob, indent=4)
    # Save model after fitting
    print('Saving model after training...')
    model.save_weights(os.path.join(network_folder_path,'trained_model_weights.h5'))
    model.save(os.path.join(network_folder_path, 'model_after_training'))

    # Plot and save metrics as figure(s)
    print('Plotting training history...')
    cnn_model.plot_and_save_metrics(training_history, folder=network_folder_path, file_prefix='figure')

    print('Predicting on validation set...')
    df_validation_w_predictions = cnn_model.prediction_on_df(model, df_validation, image_folder=image_folder)
    # Save/dump dataframe
    df_val_filename = os.path.split(dataframe_validation_file)[-1]
    df_val_filename_parts = df_val_filename.split(sep='.')
    df_val_filename_out = '.'.join(df_val_filename_parts[:-1]) + '_w_pred_'  + unique_identifier_from_args + '.pkl'
    print('Dumping predictions to file: ', df_val_filename)
    df_validation_w_predictions.to_pickle(os.path.join(network_folder_path, df_val_filename_out))

    # Create, display and save confusion matrix of validation set
    print('\nConfusion matrix from validation set:')
    labels = [key for key in labels_dict.keys()]
    CM_validation = CM.confusionmatrix(N_classes, labels)
    CM_validation.Append(df_validation['label_no'], df_validation['pred_label_no'])
    CM_validation.Save(os.path.join(network_folder_path, 'ConfMat_val.csv'), fileFormat='csv')
    print(CM_validation)
    print(labels_dict)

    print('Training and validation completed!')
    print('Network path: ', network_folder_path)

if __name__ == '__main__':
    main()