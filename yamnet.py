import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import tensorflow_ranking as tfr


import random
import csv

FFT_SIZE_IN_SECS = 0.05
HOP_LENGTH_IN_SECS = 0.01

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

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
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

def load_wav_for_map_detection(filename, offset_st, offset_ed, label):
    wav = load_wav_16k_mono(filename)
    wav = wav[offset_st*16000:offset_ed*16000]
    return wav, label

def extract_embedding_detection(wav_data, label):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
          tf.repeat(tf.expand_dims(label,axis=0),num_embeddings,axis=0),)

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

def load_data(file, index_dict):
    with open(file) as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data['data'])

    filenames = df['wav']
    targets = df['labels'].map(lambda x: int(index_dict[x]))

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets))
    main_ds = main_ds.map(load_wav_for_map)
    main_ds = main_ds.map(extract_embedding).unbatch()

    return main_ds

def load_detection_data(file, index_dict, dir):
    xs = []
    start_offsets = []
    end_offsets = []
    ys = []

    if dir == 'dcase':
        window_width = 2
        window_shift = 1
    else:
        window_width = 10 # dcase: 2, hiceas: 10
        window_shift = 5 # dcase: 1, hiceas: 5

    # self.data = data_json['data']
    with open(file, 'r') as fp:
        data_json = json.load(fp)
        for sample in data_json['data']:
            # data = json.loads(line)
            wav_path = sample['wav']
            length = sample['length']

            num_windows = int((length - window_width) / window_shift) + 1

            for window_id in range(num_windows):
                st, ed = window_id * window_shift, window_id * window_shift + window_width
                # offset_st, offset_ed = st * size_per_sec, ed * size_per_sec
                offset_st, offset_ed = st, ed # because we split on audio directly, no need to multiply size_per_sec
                xs.append(wav_path)
                start_offsets.append(offset_st)
                end_offsets.append(offset_ed)

                y = [0] * len(index_dict)

                for anon in sample['annotations']:
                    try:
                        label_id = int(index_dict[anon['label']])
                    except:
                        label_id = int(index_dict[str(anon['label'])])

                    if (st <= anon['st'] <= ed) or (st <= anon['ed'] <= ed):
                        denom = min(ed - st, anon['ed'] - anon['st'])
                        if denom == 0:
                            continue
                        overlap = (min(ed, anon['ed']) - max(st, anon['st'])) / denom
                        if overlap > .2:
                            y[label_id] = 1
                    if anon['st'] <= st and ed <= anon['ed']:
                        y[label_id] = 1

                ys.append(y)

    main_ds = tf.data.Dataset.from_tensor_slices((xs, start_offsets, end_offsets, ys))
    main_ds = main_ds.map(load_wav_for_map_detection)
    main_ds = main_ds.map(extract_embedding_detection).unbatch()

    return main_ds

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

BATCH_SIZE = 32
# dirs = ['/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_health_30min_trainBo_few_shot', '/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_health_30min_few_shot','/mnt/data/tungtran/AudioMAE/dataset/coral_sound_indo_location_30min_few_shot', '/mnt/data/tungtran/AudioMAE/dataset/coral_sound_coral_chorus_30min_few_shot']
# dirs = ['indo_health_15min_few_shot', 'indo_health_1h_few_shot','indo_location_15min_few_shot', 'indo_location_1h_few_shot', 'coral_chorus_15min_few_shot', 'coral_chorus_30min_few_shot', 'coral_chorus_1h_few_shot']
# dirs = ['indo_health_30min_few_shot_v2']
# dirs = ['dataset/coral_sound_' + dir for dir in dirs]
dirs = ['hiceas']
dirs = ['dataset/' + dir for dir in dirs]

seeds = [3]
for seed in seeds:
    for dir in dirs:
        DIR = dir
        set_seed(seed)
        dataset_dir = DIR.split('/')[-1]

        index_dict = make_index_dict(f'{DIR}/class_labels_indices.csv')
        train_ds = load_detection_data(f'{DIR}/train.json', index_dict, dataset_dir).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = load_detection_data(f'{DIR}/val.json', index_dict, dataset_dir).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        test_ds = load_detection_data(f'{DIR}/test.json', index_dict, dataset_dir).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        if dataset_dir == 'dcase':
            NUM_CLASSES = 18
        else:
            NUM_CLASSES = 1 # dcase: 18, hiceas: 1

        my_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                                  name='input_embedding'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES)
        ], name='my_model')

        print(my_model.summary())

        # For classification task:
        # my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # For multi-label task:
        my_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         optimizer="adam",
                         metrics=[tfr.keras.metrics.MeanAveragePrecisionMetric()])

        log_dir = "logs/" + DIR.split('/')[-1] + f'_seed{seed}_yamnet_gpu'
        print(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_mean_average_precision_metric', save_best_only=True, mode='max')

        history = my_model.fit(train_ds,
                               epochs=60,
                               validation_data=val_ds,
                               callbacks=[tensorboard_callback, checkpoint])

        # 5. Select Best Epoch
        best_epoch = tf.argmax(history.history['val_mean_average_precision_metric']) + 1  # +1 because epochs are 1-indexed

        # 6. Evaluate on Test Set
        my_model.load_weights('best_model.h5')  # Load weights of the best model
        test_loss, test_accuracy = my_model.evaluate(test_ds)

        # Log to TensorBoard
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('test_loss', test_loss, step=0)
            tf.summary.scalar('test_accuracy', test_accuracy, step=0)

        print()
