import ast
import os

# setting_keywords = 'indo_health_30min_trainBo_few_shot_lr4e-3_freezeTrue'
# setting_keywords = 'indo_health_30min_few_shot_lr4e-3_freezeFalse'
# setting_keywords = 'indo_location_30min_few_shot_lr4e-3_freezeFalse'
# second_keyword = '_chorus_pretrained'
<<<<<<< HEAD
# setting_keywords = 'coral_chorus_30min_few_shot_lr4e-3_freezeTrue'
# second_keyword = 'scratch'
# second_keyword = '_indo_pretrained'
setting_keywords = 'watkins_lr4e-3_freezeTrue'
second_keyword = 'audioset'
# second_keyword = 'scratch'
split = 'test'
metrics = ['acc1', 'acc2', 'mAP', 'mAUC', 'f1', 'precision', 'recall']
=======
# second_keyword = 'scratch'
# setting_keywords = 'coral_chorus_30min_few_shot_lr4e-3_freezeFalse'
# second_keyword = '_indo_pretrained'
setting_keywords = 'hiceas_lr8e-3'
# setting_keywords = 'dcase'
second_keyword = '0_no_norm_audioset'
# second_keyword = '0_australia'
# second_keyword = '_5e_australia'
# second_keyword = '_weight_balancer_australia'
# second_keyword = '_weight_balancer_no_norm_australia_large'
# second_keyword = '_weight_balancer_no_norm_australia'
# second_keyword = '_weight_balancer_no_norm_scratch'
# second_keyword = '_weight_balancer_no_norm_audioset'
# second_keyword = '_audioset_pretrained'
split = 'real_test'
# metrics = ['acc1', 'acc2', 'mAP', 'mAUC', 'f1', 'precision', 'recall', 'beans_mAP']
metrics = ['beans_mAP']
>>>>>>> 2b0bdea8118f070159225a77ceb1ec7ea64c5e7e

folder_list = [folder for folder in os.listdir('logs') if setting_keywords in folder and folder.endswith(second_keyword)]
results = {folder: {} for folder in folder_list}
print(setting_keywords, second_keyword, split, results)

for folder in folder_list:
    with open('logs/' + folder + '/log.txt') as f:
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
        ax.plot(list(results[folder].keys()), [results[folder][epoch][f'{split}_' + metric] for epoch in results[folder].keys()], label=folder)
    # plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(metric)
    plt.savefig('plots/' + split + '_' + metric + '.png')
    plt.close()

# assert len(results.keys()) == 3, "Not all runs have finished training."

# for i in range(1, len(folder_list)):
#     assert len(results[folder_list[0]]) == len(results[folder_list[i]]), f"Run {i} has not finished training."

test_best_index = 0
for split in ['test', 'real_test']:
    print("SPLIT: ", split)
    # compute average results:
    for metric in metrics:
        print(metric)
        avg = []
        for epoch in results[folder_list[0]].keys():
            avg.append(np.mean([results[folder][epoch][f'{split}_' + metric] for folder in folder_list]))
        # for folder in folder_list:
        #     print(folder, np.mean([results[folder][epoch]['test_' + metric] for epoch in results[folder].keys()]))
        max_value_index = np.argmax(avg)
        std = np.std([results[folder][max_value_index][f'{split}_' + metric] for folder in folder_list])
        print(f"{np.max(avg):.2f} +- {std:.2f}",  np.argmax(avg))
        if split == 'test':
            test_best_index = max_value_index
        else:
            print("BEST on real test: ", f"{avg[test_best_index]:.3f} +- {std:.2f}")
