# %%
from datetime import time
import pickle
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy
import math
from scipy.stats import skew, kurtosis
import argparse

# %%
# ACC:accelerometer data
parser = argparse.ArgumentParser(description='the parameter is the subject number')

parser.add_argument("-n", "--number", type=int, help='subject number to indicate the file name')
args = parser.parse_args()

if args.number:
    print(args.number)
    file_num = args.number
else:
    file_num = 15

subject_name = 'S'+str(file_num)

p = Path("../PPG_FieldStudy/"+subject_name + "/" + subject_name + ".pkl")
with p.open(mode='rb') as f:
    # show = pickle.load(f)
    pkl_data = pickle.load(f, encoding='latin-1')

# %%
# Signal processing 
def bandpass (signal, fs):
    pass_band=(0.5, 4)
    b, a = scipy.signal.butter(2, pass_band, btype='bandpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

# %%
# Data Statistics

# Capture Empatica E4
wrist_ACC = pkl_data['signal']['wrist']['ACC'] # 32 Hz
wrist_BVP = pkl_data['signal']['wrist']['BVP'] # 64 Hz
wrist_EDA = pkl_data['signal']['wrist']['EDA'] # 4 Hz
wrist_TEMP = pkl_data['signal']['wrist']['TEMP'] # 4 Hz

# Capture RespiBAN
chest_ACC = pkl_data['signal']['chest']['ACC'] # 700 Hz
chest_ECG = pkl_data['signal']['chest']['ECG'] # 700 Hz
chest_resp = pkl_data['signal']['chest']['Resp'] # 700 Hz

# Activity
activity = pkl_data['activity']

# Heart rate: the segmentation time is 2s
heart_rate_original = np.array(pkl_data['label'])
heart_rate = heart_rate_original.repeat(2)
# The first element should extend to 6s since the first segmentation is 8s
heart_rate = np.insert(heart_rate, -1, np.ones((7,))*heart_rate[-1])


# print('len wrist_ACC : ' + str(len(wrist_ACC)))
# print('len wrist_BVP : ' + str(len(wrist_BVP)))
# print('len wrist_EDA : ' + str(len(wrist_EDA)))
# print('len wrist_TEMP : ' + str(len(wrist_TEMP)))
# print("---------Empatica E4---------")
# print('len chest_ACC : ' + str(len(chest_ACC)))
# print('len chest_ECG : ' + str(len(chest_ECG)))
# print('len chest_resp : ' + str(len(chest_resp)))

# %%
# Data Transformation
def standardize_to_700hz(sensor_values, frequency):
    ratio = 700/frequency
    new_len = int(len(sensor_values) * ratio) # len of the standardized list
    if (len(sensor_values.shape) > 1):
        dim_value = sensor_values.shape[1]
        sensor_values_700hz = np.zeros((new_len, dim_value))
    else:
        sensor_values_700hz = np.zeros((new_len, )) 
    for i in range(new_len):
        sensor_values_700hz[i] = sensor_values[int(i/ratio)]
    return sensor_values_700hz

wrist_ACC_700hz = standardize_to_700hz(wrist_ACC, 32)
wrist_BVP_700hz = standardize_to_700hz(wrist_BVP, 64)
wrist_EDA_700hz = standardize_to_700hz(wrist_EDA, 4)
wrist_TEMP_700hz = standardize_to_700hz(wrist_TEMP, 4)

wrist_acc_x_flat = wrist_ACC_700hz[:,0]
wrist_acc_y_flat = wrist_ACC_700hz[:,1]
wrist_acc_z_flat = wrist_ACC_700hz[:,2]

activity_700hz = standardize_to_700hz(activity, 4)
# signal_df = pd.DataFrame(data["signal"])
# %% 

# Data Visualization
chest_ecg_flat = chest_ECG.flatten()
chest_acc_x_flat = chest_ACC[:,0]
chest_acc_y_flat = chest_ACC[:,1]
chest_acc_z_flat = chest_ACC[:,2]
chest_resp_flat = chest_resp.flatten()

wrist_BVP_flat = wrist_BVP_700hz.flatten()
wrist_BVP_flat = (wrist_BVP_flat - wrist_BVP_flat.mean()) / wrist_BVP_flat.std()
activity_flat = activity_700hz.flatten()

# plt.plot(chest_ecg_flat[0:10000])
# plt.plot(chest_acc_x_flat[0:10000])
# plt.plot(chest_acc_y_flat[0:10000])
# plt.plot(chest_acc_z_flat[0:10000])
# plt.plot(chest_resp_flat[0:10000])
# %%
# DataFrame begins
subject = pkl_data['subject']
age = pkl_data['questionnaire']['AGE']
gender = pkl_data['questionnaire']['Gender']
height = pkl_data['questionnaire']['HEIGHT']
weight = pkl_data['questionnaire']['WEIGHT']
skin = pkl_data['questionnaire']['SKIN']
sport = pkl_data['questionnaire']['SPORT']


df = {'activity': activity_flat, \
        'wrist_bvp': wrist_BVP_flat, \
        #'wrist_temp':wrist_TEMP_700hz, \
        'chest_acc_x':chest_acc_x_flat, \
        'chest_acc_y':chest_acc_y_flat, \
        'chest_acc_z':chest_acc_z_flat, \
        'chest_ecg': chest_ecg_flat, \
        'chest_resp':chest_resp_flat,
        'wrist_acc_x': wrist_acc_x_flat,
        'wrist_acc_y': wrist_acc_y_flat,
        'wrist_acc_z': wrist_acc_z_flat,
        }
# %%
df_data = pd.DataFrame(df)

# update time stamp
df_data['time'] = np.array(df_data.index) / 700
df_data['hr'] = heart_rate[df_data['time'].astype('int')]

# %%
from dtw import dtw
def norm(x, y):
    return math.fabs(x - y)

def update_act(feature_df, segment, col_name):
    segment = segment['activity']
    values, counts = np.unique(segment, return_counts=True)
    activity_value = values[np.argmax(counts)]
    feature_df = feature_df.append({col_name:activity_value}, ignore_index=True) 
    return feature_df

def update_f1(feature_df, segment, col_name):
    f1 = np.mean(segment['wrist_bvp'])
    feature_df.iloc[-1][col_name] = f1
    return feature_df

def update_f2(feature_df, segment, col_name):
    f2 = np.std(segment['wrist_bvp'])
    feature_df.iloc[-1][col_name] = f2
    return feature_df

def update_f3(feature_df, segment, col_name):
    f3 = np.nanmax(segment['wrist_bvp'])
    feature_df.iloc[-1][col_name] = f3
    return feature_df


# Minimum value based on ppt
def update_f4(feature_df, segment, col_name):
    f4 = np.nanmin(segment['wrist_bvp'])
    feature_df.iloc[-1][col_name] = f4
    return feature_df


# Maximum position
def update_f5(feature_df, segment, col_name):
    temp = segment['wrist_bvp']
    length = len(temp)
    f5 = np.argmax(temp)/length
    feature_df.iloc[-1][col_name] = f5
    return feature_df


# Minimum position
def update_f6(feature_df, segment, col_name):
    temp = segment['wrist_bvp']
    length = len(temp)
    f6 = np.argmin(temp)/length
    feature_df.iloc[-1][col_name] = f6
    return feature_df

def update_f7(feature_df, segment, col_name):
    f7 = segment['hr'].mean()
    feature_df.iloc[-1][col_name] = f7
    return feature_df 


# Fisher-Pearson coefficient skewness
def update_f8(feature_df, segment, col_name):
    temp = segment['wrist_bvp']
    f8 = skew(temp)
    feature_df.iloc[-1][col_name] = f8
    return feature_df


# Kurtosis of the PPG segment
def update_f9(feature_df, segment, col_name):
    temp = segment['wrist_bvp']
    f9 = kurtosis(temp)
    feature_df.iloc[-1][col_name] = f9
    return feature_df

def get_time_list(time_step, time_shift, st_time, end_time):
    arr = np.array([st_time, st_time+time_step])
    st_time = st_time + time_shift

    while (st_time + time_step) <= end_time:
        temp_end = st_time + time_step
        arr = np.vstack([arr, np.array([st_time, temp_end])])
        st_time = st_time + time_shift
    return arr

def calculate_dtw(time_list, df_data=None):
    total_len = len(time_list)
    dtw_matrix = np.zeros((total_len, total_len))
    for idx, item in enumerate(time_list):
        left_num = total_len - (idx + 1)
        current_segment = df_data.loc[(df_data['time'] >= item[0]) & (df_data['time'] < item[1]),:]
        for j in range(left_num):
            nxt_id = j+1
            compare_time = time_list[nxt_id]
            print(compare_time[0])
            compare_segment = df_data.loc[(df_data['time'] >= compare_time[0]) & (df_data['time'] < compare_time[1]),:]
            dtw_return = dtw(current_segment.loc[:,'wrist_bvp'].values, compare_segment.loc[:,'wrist_bvp'].values, dist=norm)
            dtw_matrix[idx][j] = dtw_return[0] 

    return dtw_matrix

# dtw_matrix = calculate_dtw(time_list, df_data)
# %%


# print(feature_df)
def get_featuredf(df_data):
    # Feature Extraction
    time_step, time_shift, st_time, end_time = 8, 2, 0, math.ceil(max(df_data['time']))
    # time_step, time_shift, st_time, end_time = 8, 2, 0, 100
    # time_list = get_time_list(time_step, time_shift, st_time, end_time)
    col_names = ['activity', 'f1:mean', 'f2:std', 'f3:max', 'f4:min',
                'f5:max_position', 'f6:min_position', 'f7:hr',
                'f8:skewness', 'f9:kurtosis']
    feature_df = pd.DataFrame(columns=col_names)

    while (st_time + time_step) <= end_time:
        segment = df_data.loc[(df_data['time'] >= st_time) & (df_data['time'] < (st_time + time_step)),:]

        # update activity
        feature_df = update_act(feature_df, segment, col_names[0])
        
        # update f1
        feature_df = update_f1(feature_df,segment, col_names[1])

        # update f2
        feature_df = update_f2(feature_df,segment, col_names[2])

        # calculate the rest features
        feature_df = update_f3(feature_df, segment, col_names[3])
        feature_df = update_f4(feature_df, segment, col_names[4])
        feature_df = update_f5(feature_df, segment, col_names[5])
        feature_df = update_f6(feature_df, segment, col_names[6])
        feature_df = update_f7(feature_df, segment, col_names[7]) 
        feature_df = update_f8(feature_df, segment, col_names[8])
        feature_df = update_f9(feature_df, segment, col_names[9])
        

        st_time = st_time + time_shift
    
    return feature_df

feature_df = get_featuredf(df_data)
feature_df.to_csv(subject_name + ".csv")

# %%

# Visualization part
'''
df_data_temp = df_data[df_data['time'] < 20]
plt.plot(df_data_temp['time'], df_data_temp['chest_ecg'], label='ecg amplitude')
plt.title("ECG signal in the time-domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

# %%
plt.plot(df_data_temp['time'], df_data_temp['wrist_acc_x'], label="x axis accelerator")
plt.plot(df_data_temp['time'], df_data_temp['wrist_acc_y'], label='ecg amplitude')
plt.title("ECG signal in the time-domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
# %%


freq_wrist_bvp = np.fft.rfftfreq(len(wrist_BVP_flat), 1/700)
fft_wrist_bvp = np.fft.rfft(wrist_BVP_flat)

plt.plot(freq_wrist_bvp[0:100000], np.abs(fft_wrist_bvp[0:100000])/np.max(np.abs(fft_wrist_bvp)), label="PPG signal")
plt.title("PPG signal in the frequency-domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()


# freq_wrist_acc_x = np.fft.rfftfreq(len(wrist_acc_x_flat), 1/700) 
# freq_wrist_acc_y = np.fft.rfftfreq(len(wrist_acc_y_flat), 1/700) 
# freq_wrist_acc_z = np.fft.rfftfreq(len(wrist_acc_z_flat), 1/700) 

# fft_wrist_acc_x = np.fft.rfft(wrist_acc_x_flat) 
# fft_wrist_acc_y = np.fft.rfft(wrist_acc_x_flat) 
# fft_wrist_acc_z = np.fft.rfft(wrist_acc_x_flat) 
# plt.plot(freq_wrist_acc_x[0:100000], np.abs(freq_wrist_acc_x[0:100000])/np.max(np.abs(freq_wrist_acc_x)), label="ACC_X")

freq_chest_ecg = np.fft.rfftfreq(len(chest_ecg_flat), 1/700)
fft_chest_ecg = np.fft.rfft(chest_ecg_flat)

plt.plot(freq_chest_ecg[0:1000000], np.abs(chest_ecg_flat[0:1000000])/np.max(np.abs(fft_chest_ecg)), label="PPG signal")
plt.title("ECG signal in the frequency-domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()
# %%
# plot the heart rate and the activity

# get the activity seperation line
t = abs(np.diff(df_data['activity']))
seps_array = np.where(t>0)[0]
print(seps_array)
fig, ax1 = plt.subplots(1,1, figsize=(9, 5))


ax1.set_xlabel('Time(s)')

color = 'b'
ax1.set_ylabel('Heart rate in bpm')
ax1.plot(df_data['time'], df_data['hr'], color=color)
ax1.tick_params(axis='y')
for i in seps_array:
    x_axis = df_data.loc[i]['time']
    ax1.axvline(x_axis,ls='--',color='r')
# ax1.fill_between(df_data['time'], 0, df_data['activity'], color=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'b'
# ax2.set_ylabel('heart rate', color=color)  # we already handled the x-label with ax1
# ax2.plot(df_data['time'], df_data['hr'], color=color)

# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Heart Rate and Activity Labels')
plt.show()
# %%
plt.style.use("seaborn")
df_data_temp = df_data[(df_data['time'] > 20) & (df_data['time'] < 40)]
plt.plot(df_data_temp['time'], df_data_temp['wrist_bvp'], label="PPG amplitude")
plt.title("PPG signal in the time-domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

filtered_wrist_bvp = bandpass(df_data_temp['wrist_bvp'], 700)
plt.plot(filtered_wrist_bvp)
'''
