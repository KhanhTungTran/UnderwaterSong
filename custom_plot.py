import ast
import os

setting_keywords = 'coral_chorus_2hours_few_shot_lr1e-3'
type = 'finetune'
split = 'test'
metrics = ['acc1', 'acc2', 'mAP', 'mAUC', 'f1', 'precision', 'recall']

folder_list = [folder for folder in os.listdir() if setting_keywords in folder and type in folder]
results = {folder: {} for folder in folder_list}

for folder in folder_list:
    with open(folder + '/log.txt') as f:
        log = f.readlines()
    for line in log:
        curr_dct = ast.literal_eval(line)
        results[folder][curr_dct['epoch']] = curr_dct

import matplotlib.pyplot as plt
import numpy as np

# plot the results:
for metric in metrics:
    fig = plt.figure(figsize=(18, 6), dpi=80)
    ax = plt.subplot(111)
    for folder in folder_list:
        ax.plot(list(results[folder].keys()), [results[folder][epoch]['test_' + metric] for epoch in results[folder].keys()], label=folder)
    # plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(metric)
    plt.savefig('plots/' + metric + '.png')
    plt.close()
