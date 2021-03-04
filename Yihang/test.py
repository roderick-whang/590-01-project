# %%
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from pathlib import Path

# %%
p = Path("./S7.pkl")
with p.open(mode='rb') as f:
    # show = pickle.load(f)
    data = pickle.load(f, encoding='latin-1')
# %%
# Formatting activity property
activities = []
for i in range(len(data['activity'])):
    activities.append(data['activity'][i][0])

# Formatting label property
new_label = []
for i in range(len(data['label'])):
    for x in range(8):
        new_label.append(data['label'][i])

print(len(new_label))
while len(new_label) < len(data['activity']):
    new_label.append(data['label'][i])
    
# Formatting signal.wrist property 
# - ACC : conversion de 32 Hz en 4Hz
signal_wrist_ACC_3D = []
signal_wrist_ACC_3D.append([]) # 3 dimensions
signal_wrist_ACC_3D.append([])
signal_wrist_ACC_3D.append([])
i = 0
dataSignalWristACC = data['signal']['wrist']['ACC']
while i < len(dataSignalWristACC) :
    signal_wrist_ACC_3D[0].append(dataSignalWristACC[i][0])
    signal_wrist_ACC_3D[1].append(dataSignalWristACC[i][1])
    signal_wrist_ACC_3D[2].append(dataSignalWristACC[i][2])
    i = i + 8
    
# - BVP : conversion de 64 Hz en 4 Hz
signal_wrist_BVP = []
i = 0
dataSignalWristBVP = data['signal']['wrist']['BVP']
while i < len(dataSignalWristBVP) :
    signal_wrist_BVP.append(dataSignalWristBVP[i][0])
    i = i + 16

# Formatting signal.chest property
# - conversion de 700 Hz en 4 Hz
# - exclude EDA, EMG and Temp which include dummy data (as the readme file say)
signal_chest = {}
signal_chest['ACC_3D'] = []
signal_chest['ACC_3D'].append([]) # 3 dimensions
signal_chest['ACC_3D'].append([])
signal_chest['ACC_3D'].append([])
signal_chest['ECG'] = []
signal_chest['Resp'] = []
i = 0
dataSignalChest = data['signal']['chest']
while i < len (dataSignalChest['ACC']): # each channel has the same length
    signal_chest['ACC_3D'][0].append(dataSignalChest['ACC'][i][0])
    signal_chest['ACC_3D'][1].append(dataSignalChest['ACC'][i][1])
    signal_chest['ACC_3D'][2].append(dataSignalChest['ACC'][i][2])
    signal_chest['ECG'].append(dataSignalChest['ECG'][i][0])
    signal_chest['Resp'].append(dataSignalChest['Resp'][i][0])
    i = i + 175

if data["questionnaire"]["Gender"].strip() == 'm':
    gender = 1
else:
    gender = 0

# Create dataframe
df = pd.DataFrame({
    "subject_ID": [int(7)*200 for i in range(len(data['activity']))],
    "activity": activities,
    "label": new_label,
    "wrist_ACC1": signal_wrist_ACC_3D[0],
    "wrist_ACC2": signal_wrist_ACC_3D[1],
    "wrist_ACC3": signal_wrist_ACC_3D[2],
    "wrist_BVP": signal_wrist_BVP,
    "wrist_EDA": [x[0] for x in data['signal']['wrist']['EDA']],
    "wrist_TEMP": [x[0] for x in data['signal']['wrist']['TEMP']],
    "chest_ACC1": signal_chest['ACC_3D'][0],
    "chest_ACC2": signal_chest['ACC_3D'][1],
    "chest_ACC3": signal_chest['ACC_3D'][2],
    "chest_ECG": signal_chest['ECG'],
    "chest_Resp": signal_chest['Resp'],
    "age": [data["questionnaire"]["AGE"] for i in range(len(data['activity']))],
    "gender": [gender for i in range(len(data['activity']))],
    "height": [data["questionnaire"]["HEIGHT"] for i in range(len(data['activity']))],
    "skin": [data["questionnaire"]["SKIN"] for i in range(len(data['activity']))],
    "sport": [data["questionnaire"]["SPORT"] for i in range(len(data['activity']))],
    "weight": [data["questionnaire"]["WEIGHT"] for i in range(len(data['activity']))]
})

# Waiting the execution, you can visualize the heart rate by subject
fig, ax1 = plt.subplots()

ax1.set_xlabel('time')

color = 'c'
ax1.set_ylabel('activity', color=color)
ax1.plot(df.index, df.activity, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.fill_between(df.index, 0, df.activity, color=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'b'
ax2.set_ylabel('Heart Rate in bpm', color=color)  # we already handled the x-label with ax1
ax2.plot(df.index, df.label, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('')
plt.show()

details = data['questionnaire']
details['mean_heart_rate'] = np.mean(new_label)
details['min_heart_rate'] = np.min(new_label)
details['max_heart_rate'] = np.max(new_label)
        


# %%
