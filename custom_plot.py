import ast
import os

# setting_keywords = 'indo_health_30min_trainBo_few_shot_lr4e-3_freezeTrue'
# setting_keywords = 'indo_health_30min_few_shot_lr4e-3_freezeTrue'
# setting_keywords = 'indo_location_30min_few_shot_lr4e-3_freezeTrue'
# second_keyword = '_chorus_pretrained'
# second_keyword = 'scratch'
# setting_keywords = 'coral_chorus_30min_few_shot_lr4e-3_freezeTrue'
# second_keyword = '_indo_pretrained'
# second_keyword = 'scratch'
setting_keywords = 'watkins_lr4e-3_freezeTrue'
second_keyword = '_australia'
split = 'test'
metrics = ['acc1', 'acc2', 'mAP', 'mAUC', 'f1', 'precision', 'recall']

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

# # plot the results:
# for metric in metrics:
#     fig = plt.figure(figsize=(18, 6), dpi=80)
#     ax = plt.subplot(111)
#     for folder in folder_list:
#         ax.plot(list(results[folder].keys()), [results[folder][epoch]['test_' + metric] for epoch in results[folder].keys()], label=folder)
#     # plt.legend()
#     # Shrink current axis by 20%
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#     # Put a legend to the right of the current axis
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.title(metric)
#     plt.savefig('plots/' + metric + '.png')
    # plt.close()

assert len(results.keys()) == 3, "Not all runs have finished training."

for i in range(1, len(folder_list)):
    assert len(results[folder_list[0]]) == len(results[folder_list[i]]), f"Run {i} has not finished training."

# compute average results:
for metric in metrics:
    print(metric)
    avg = []
    for epoch in results[folder_list[0]].keys():
        avg.append(np.mean([results[folder][epoch]['test_' + metric] for folder in folder_list]))
    # for folder in folder_list:
    #     print(folder, np.mean([results[folder][epoch]['test_' + metric] for epoch in results[folder].keys()]))
    max_value_index = np.argmax(avg)
    std = np.std([results[folder][max_value_index]['test_' + metric] for folder in folder_list])
    print(f"{np.max(avg):.2f} +- {std:.2f}",  np.argmax(avg))
