import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
import skimage.transform
import tensorflow as tf
import tqdm

class dataset_sample_generator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=1, image_folder='.', shuffle_end_of_epoch=True, name='', image_size=None):
        self.df = df
        self.indices = np.arange(len(self.df))
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.shuffle_end_of_epoch = shuffle_end_of_epoch
        if (self.shuffle_end_of_epoch):
            np.random.shuffle(self.indices)
        self.name = name
        self.image_size = image_size

    def __len__(self):
        return (np.ceil(len(self.df) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        df_batch = self.df.iloc[batch_indices,:]

        if self.image_size:
            batch_x = np.array([skimage.transform.resize(skimage.io.imread(os.path.join(self.image_folder, row['folder'], row['image'])), self.image_size) for r,row in df_batch.iterrows()]) #, dtype=object)
            # batch_x = np.concatenate([skimage.transform.resize(skimage.io.imread(os.path.join(self.image_folder, row['folder'], row['image'])), self.image_size) for r,row in df_batch.iterrows()], axis=0)
        else:
            batch_x = np.array([skimage.io.imread(os.path.join(self.image_folder, row['folder'], row['image'])) for r, row in df_batch.iterrows()]) #, dtype=object)
            # batch_x = np.concatenate([skimage.io.imread(os.path.join(self.image_folder, row['folder'], row['image'])) for r, row in df_batch.iterrows()], axis=0)
        # batch_x = tf.keras.preprocessing.image.random_shear(batch_x, intensity=5, fill_mode='reflect')
        # print(batch_x.shape)
        batch_y = np.vstack(df_batch['label_one_hot']).astype('float32')
        return batch_x, batch_y

    def on_epoch_end(self):
        # print(self.name + ': End of epoch')
        if self.shuffle_end_of_epoch:
            np.random.shuffle(self.indices)
            # print(self.name + ': Do the shuffle...')

# def data_augmentation(x):
#     x = tf.keras.preprocessing.image.random_shear(x, intensity=5, fill_mode='reflect')
#     x = tf.keras.preprocessing.image.random_brightness(x, brightness_range=(0.8, 1.2))
#     # x = tf.keras.layers.experimental.preprocessing.RandomFlip()
#     return x

def build_model(input_shape, N_classes, basenet='ResNet50V2', weights='imagenet', pooling='avg', fine_tune_only=False, **kwargs):
    ## Setup network
    i = tf.keras.layers.Input(input_shape, dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    # TODO: Preprocessing

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.2),
            # tf.keras.layers.experimental.preprocessing.RandomCrop(height=input_shape[0], width=input_shape[0])
        ]
    )
    x = data_augmentation(x)

    if basenet.lower() == 'ResNet50V2'.lower():
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        model = tf.keras.applications.ResNet50V2(include_top=False, weights=weights, input_shape=[input_shape[0], input_shape[0], input_shape[2]], pooling=pooling, classes=N_classes, **kwargs)
    elif basenet.lower() == 'MobileNetV2'.lower():
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=[input_shape[0], input_shape[0], input_shape[2]], pooling=pooling, classes=N_classes, **kwargs)
    elif basenet.lower() == 'MobileNetV3Large'.lower():
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        model = tf.keras.applications.MobileNetV3Large(include_top=False, weights=weights, input_shape=[input_shape[0], input_shape[0], input_shape[2]], pooling=pooling, classes=N_classes, **kwargs)
    elif basenet.lower() == 'MobileNetV3Small'.lower():
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        model = tf.keras.applications.MobileNetV3Small(include_top=False, weights=weights, input_shape=[input_shape[0], input_shape[0], input_shape[2]], pooling=pooling, classes=N_classes, **kwargs)
    else:
        raise ValueError('Unknown base network: ' + basenet)

    if fine_tune_only:
        for layer in model.layers:
            layer.trainable = False
    model = tf.keras.Sequential([model, tf.keras.layers.Dense(N_classes)])
    x = model(x)
    return tf.keras.Model(inputs=[i], outputs=[x])

def load_model(filepath):
    return tf.keras.models.load_model(filepath)


def setup_optimizer(optimizer_name, optimizer_params_dict):
    if optimizer_name == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(**optimizer_params_dict)
    elif optimizer_name == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(**optimizer_params_dict)
    elif optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(**optimizer_params_dict)
    elif optimizer_name == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(**optimizer_params_dict)
    elif optimizer_name == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(**optimizer_params_dict)
    elif optimizer_name == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(**optimizer_params_dict)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(**optimizer_params_dict)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**optimizer_params_dict)
    else:
        raise ValueError('Unknown name of optimizer: ' + optimizer_name)

    return optimizer

def setup_callbacks(network_folder_path):
    callbacks = []
    csv_logger_callback = tf.keras.callbacks.CSVLogger(os.path.join(network_folder_path, 'training.log'))
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(network_folder_path, 'best_model_checkpoint'), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(network_folder_path, 'logs'), histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=1)
    callbacks = [csv_logger_callback, checkpoint_callback, tensorboard_callback]
    return callbacks

def setup_loss_func(from_logits=True):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)

def prediction_on_df(model, df, image_folder='.'):
    validation_predictions = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='Predicting on dataframe'):
        I = skimage.io.imread(os.path.join(image_folder, row['folder'], row['image']))
        val_pred = model.predict(x=np.expand_dims(I,axis=0), verbose=0)
        validation_predictions.append(val_pred)
    df['predictions'] = validation_predictions
    df['pred_label_no'] = np.argmax(np.asarray(validation_predictions).squeeze(), axis=1)
    return df

def plot_and_save_metrics(training_history, folder, file_prefix='figure'):
    metrics = np.unique([key[4:] if key[0:3] == 'val' else key for key in training_history['history'].keys()])
    for metric in metrics:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1,1,1)
        L_train = ax.plot(training_history['epoch'], training_history['history'][metric],'o-')
        L_train[0].set_label('Train')
        if 'val_' + metric in training_history['history'].keys():
            L_val = ax.plot(training_history['epoch'], training_history['history']['val_' + metric],'o-')
            L_val[0].set_label('Validation')
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.set_ylim([np.minimum(0.0, ylim[0]), np.maximum(1.0, ylim[1])])
        ax.set_xlim([np.minimum(0.0, xlim[0]), np.maximum(1.0, xlim[1])])
        ax.set_xlabel('Epoch no.')
        ax.set_ylabel(metric.title())
        ax.set_title(metric.title())
        ax.legend(loc='upper left')
        # ax.
        fig.savefig(os.path.join(folder, file_prefix + '_' + metric + '.png'), format='png')
