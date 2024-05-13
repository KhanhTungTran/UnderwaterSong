import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


import random
import os
import librosa

tf.config.run_functions_eagerly(True)
# tf.enable_eager_execution() 

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Utility functions for loading audio files and making sure the sample rate is correct.

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    # wav, sample_rate = librosa.load(filename, sr=None)
    # wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    # #pad to minimum length
    # if tf.shape(wav)[0] < 24000:
    #     wav = tf.pad(wav, [[0, 24000 - tf.shape(wav)[0]]])
    return wav

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
  print(name)
print('...')

import json
import csv

def load_wav_for_map(filename, label):
  return load_wav_16k_mono(filename), int(label)

def extract_embedding(wav_data, label):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),)

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


# def create_generator(list_of_arrays):
#     for i in list_of_arrays:
#         yield i

# def add_label(x):
#     global
#     ##use x to derive additional columns u want. Set the shape as well
#     y = {}
#     y.update(x)
#     y['new1'] = new1
#     y['new2'] = new2
#     return y

def load_data(file, index_dict):
    with open(file) as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data['data'])

    filenames = df['wav']
    targets = df['labels'].map(lambda x: int(index_dict[x]))

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets))
    main_ds = main_ds.map(load_wav_for_map)
    # wavs = [load_wav_16k_mono(filename) for filename in filenames]
    # wavs = [tf.convert_to_tensor(wav, dtype=tf.float32) for wav in wavs]
    # wavs = [tf.RaggedTensor.from_tensor(wav) for wav in wavs]
    # dataset = tf.data.Dataset.from_generator(lambda: create_generator(wavs),output_types= tf.float32, output_shapes=(None))
    # main_ds = tf.data.Dataset.from_tensor_slices((wavs, targets))
    main_ds = main_ds.map(extract_embedding).unbatch()

    return main_ds

BATCH_SIZE = 32
# dirs = ['/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_health_30min_trainBo_few_shot', '/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_health_30min_few_shot','/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_location_30min_few_shot', '/mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot']
# dirs = ['indo_health_15min_few_shot', 'indo_health_1h_few_shot','indo_location_15min_few_shot', 'indo_location_1h_few_shot', 'coral_chorus_15min_few_shot', 'coral_chorus_30min_few_shot', 'coral_chorus_1h_few_shot']
# dirs = ['indo_health_30min_few_shot_v2']
# dirs = ['dataset/coral_sound_' + dir for dir in dirs]

dirs = ['watkins_16bit']
dirs = ['dataset/' + dir for dir in dirs]
seeds = [2,3,1]
for seed in seeds:
    for dir in dirs:
        DIR = dir
        set_seed(seed)
        index_dict = make_index_dict(f'{DIR}/class_labels_indices.csv')
        for d in ['train', 'val', 'test']:
            with open(f'dataset/watkins_16bit/{d}.json') as f:
                data = json.load(f)
            df = pd.DataFrame.from_dict(data['data'])
            filenames = df['wav']
            for filename in filenames:
                # print(filename)
                file_contents = tf.io.read_file(filename)
                try:
                    wav, sample_rate = tf.audio.decode_wav(
                        file_contents,
                        desired_channels=1)
                except Exception as e:
                    print(filename)
                    # os.system(f'ffmpeg -i {filename} -acodec pcm_s16le {filename}' + '.temp.wav')
                    # os.system(f'ffmpeg -i {filename} -ar 16000 {filename}' + '.temp.wav')
                    # os.system(f'rm {filename}')
                    # os.system(f'mv {filename}.temp.wav {filename}')
        val_ds = load_data(f'{DIR}/val.json', index_dict).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        train_ds = load_data(f'{DIR}/train.json', index_dict).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_ds = load_data(f'{DIR}/test.json', index_dict).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        NUM_CLASSES = 31

        my_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                                  name='input_embedding'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES)
        ], name='my_model')
        
        print(my_model.summary())
        
        my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         optimizer="adam",
                         metrics=['accuracy'])
        
        # callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
        #                                             patience=3,
        #                                             restore_best_weights=True)
        log_dir = "logs/" + DIR.split('/')[-1] + f'_seed{seed}_yamnet_gpu'
        print(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
        history = my_model.fit(train_ds,
                               epochs=50,
                               validation_data=val_ds,
                               callbacks=[tensorboard_callback, checkpoint])

        # 6. Evaluate on Test Set
        my_model.load_weights('best_model.h5')  # Load weights of the best model
        test_loss, test_accuracy = my_model.evaluate(test_ds)

        # Log to TensorBoard
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('test_loss', test_loss, step=0)
            tf.summary.scalar('test_accuracy', test_accuracy, step=0)

        print()
