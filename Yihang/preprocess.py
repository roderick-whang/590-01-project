# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
# %%
# ACC:accelerometer data
p = Path("./PPG_FieldStudy/S1/S1.pkl")
with p.open(mode='rb') as f:
    # show = pickle.load(f)
    pkl_data = pickle.load(f, encoding='latin-1')

# %%


# Capture Empatica E4
wrist_ACC = pkl_data['signal']['wrist']['ACC'] # 32 Hz
wrist_BVP = pkl_data['signal']['wrist']['BVP'] # 64 Hz
wrist_EDA = pkl_data['signal']['wrist']['EDA'] # 4 Hz
wrist_TEMP = pkl_data['signal']['wrist']['TEMP'] # 4 Hz

# Capture RespiBAN
chest_ACC = pkl_data['signal']['chest']['ACC'] # 700 Hz
chest_ECG = pkl_data['signal']['chest']['ECG'] # 700 Hz
chest_resp = pkl_data['signal']['chest']['Resp'] # 700 Hz

print('len wrist_ACC : ' + str(len(wrist_ACC)))
print('len wrist_BVP : ' + str(len(wrist_BVP)))
print('len wrist_EDA : ' + str(len(wrist_EDA)))
print('len wrist_TEMP : ' + str(len(wrist_TEMP)))
print()
print('len chest_ACC : ' + str(len(chest_ACC)))
print('len chest_ECG : ' + str(len(chest_ECG)))
print('len chest_resp : ' + str(len(chest_resp)))
print()

# signal_df = pd.DataFrame(data["signal"])
# %%
chest_ecg_flat = chest_ECG.flatten()
chest_acc_x_flat = chest_ACC[:,0]
chest_acc_y_flat = chest_ACC[:,1]
chest_acc_z_flat = chest_ACC[:,2]

plt.plot(chest_ecg_flat[0:10000])
plt.plot(chest_acc_x_flat[0:10000])
plt.plot(chest_acc_y_flat[0:10000])
plt.plot(chest_acc_z_flat[0:10000])
# %%
wrist_BVP_flat = wrist_BVP.flatten()
plt.plot(wrist_BVP_flat[0:1000])
# %%
