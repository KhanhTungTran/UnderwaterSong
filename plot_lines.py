import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(8, 8))

df = pd.read_csv("plots/CRS_result_211124.csv")

linear_probing = True
# task = 'indonesia_location'
# task = 'indonesia_health'
task = 'coral_chorus_location'
if not linear_probing:
    df = df.iloc[19:49]
    df = pd.concat([df.iloc[:12], df[(df['Task']=='base') & df['Pretrained'].isin(['scratch', 'australia']) & (df['Freeze pretrained model in finetuning'] == False)]])
else:
    df = df.iloc[31:48]
    df = pd.concat([df.iloc[10:], df[(df['Task']=='base') & df['Pretrained'].isin(['scratch', 'australia']) & (df['Freeze pretrained model in finetuning'] == True)]])

model_names = df['Pretrained']
model_name_dct = {model_name: model_name for model_name in model_names}
model_name_dct['scratch'] = 'AudioMAE'
model_name_dct['australia'] = 'UnderwaterSong'

if linear_probing:
    del model_name_dct['resnet18']
    del model_name_dct['resnet50']
    del model_name_dct['resnet152']

colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(model_name_dct.keys())))))

task_dct = {'indonesia_location': 'Indonesia (loc.)', 'indonesia_health': 'Indonesia (health.)', 'coral_chorus_location': 'Coral Chorus'}
task_name = task_dct[task]
result_column_dct = {'indonesia_location': ['indo location train 5 min', 'indo location train 10 min', 'indo location train 15 min', 'Indo location clf 30 min train set', 'indo location train 45 min', 'indo location train 1h'], 'indonesia_health': ['indo health train 5 min', 'indo health train 10 min', 'indo health train 15 min', 'indo health v2', 'indo health train 45 min', 'indo health train 1h'], 'coral_chorus_location': ['coral chorus location train 5 min', 'coral chorus location train 10 min', 'coral chorus location train 15 min', 'Coral chorus location clf 30 min train set', 'coral chorus location train 45 min', 'coral chorus location train 1h']}
result_columns = result_column_dct[task]

df[df['Freeze pretrained model in finetuning'] != (not linear_probing)]

# Define the specific x-axis points as text
x_points = ['5-min', '10-min', '15-min', '30-min', '45-min', '60-min']

# Define the corresponding y-axis values (float numbers)
# y_values = [84.56, 84.67, 88.22]

# Plot the line
for model_name, display_name in model_name_dct.items():
    y_values = list(df[df['Pretrained'] == model_name][result_columns].values[0])
    y_values  = [float(y_value) for y_value in y_values]
    print(model_name, y_values)
    if display_name == 'UnderwaterSong':
        plt.plot(x_points, y_values, marker='o', linestyle='-', label=display_name, linewidth=6)
    else:
        plt.plot(x_points, y_values, marker='o', linestyle='-', label=display_name, linewidth=3)


# Set labels and title
plt.xlabel('Train size', fontsize = 20)
plt.ylabel('Accuracy', fontsize = 20)
plt.title(f'Performance on {task_name}\nwith different training set size.', fontsize = 24)

# Set custom x-axis labels
plt.xticks(x_points, fontsize=16)
plt.yticks(fontsize=16)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2, fontsize = 18)

print(f'plots/{task}_linearprobing{linear_probing}.png')
plt.savefig(f'plots/{task}_linearprobing{linear_probing}.png',bbox_inches='tight')
