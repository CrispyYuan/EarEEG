"""
Dataset Library
"""

import numpy as np
import h5py

def read_h5py(path):
    with h5py.File(path, "r") as f:
        print(f"Reading from {path} ====================================================")
        print("Keys in the h5py file : %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data1 = np.array((f[a_group_key]))
        print(f"Number of samples : {len(data1)}")
        print(f"Shape of each data : {data1.shape}")
    return data1
 

def split_data(data_list,train_list,val_list):
    data_list = np.array(data_list)
    train_data_list = data_list[train_list]
    val_data_list = data_list[val_list]
    return train_data_list, val_data_list


# def EEG_process(data,rejected_list,i=0):
#     #_,_,_,_,_,ELA,ELE,ELI,ERA,ERG,ERE,ERI,_,_,_,_,_,ELB,ELG,ELK,ERB,ERK,_ = data
#     # 因为你在预处理时，前 12 个通道完美对应了这 12 个电极，直接按顺序截取即可
#     ELA, ELE, ELI, ERA, ERG, ERE, ERI, ELB, ELG, ELK, ERB, ERK = data[:12]
#     reject = rejected_list[i]
#     ear_eeg = [ELA,ELE,ELI,ERA,ERG,ERE,ERI,ELB,ELG,ELK,ERB,ERK]
    
#     for j in range (len(reject)):
#         ear_eeg[j]=ear_eeg[j]*reject[j]
    

#     Left_ear = (ear_eeg[0] + ear_eeg[1] + ear_eeg[2] + ear_eeg[7] + ear_eeg[8] + ear_eeg[9])/np.count_nonzero([reject[0],reject[1],reject[2],reject[7],reject[8],reject[9]])   # (ELA + ELE + ELI + ELB + ELG + ELK)/6
#     Right_ear = (ear_eeg[3] + ear_eeg[4] + ear_eeg[5] + ear_eeg[6] + ear_eeg[10] + ear_eeg[11])/np.count_nonzero([reject[3],reject[4],reject[5],reject[6],reject[10],reject[11]]) # (ERA + ERG + ERE + ERI + ERB + ERK)/6
#     L_R = Left_ear - Right_ear

#     if np.count_nonzero([reject[0],reject[7]]) != 0 :
#         L_E = (ear_eeg[0]+ear_eeg[7])/np.count_nonzero([reject[0],reject[7]]) - (ear_eeg[1]+ear_eeg[2]+ear_eeg[8]+ear_eeg[9])/np.count_nonzero([reject[1],reject[2],reject[8],reject[9]]) #(ELA + ELB)/2 - (ELE + ELI + ELG + ELK)/4
#     else:
#         L_E = np.zeros(data.shape[1])
#     if np.count_nonzero([reject[3],reject[10]]) != 0 :
#         R_E = (ear_eeg[3]+ear_eeg[10])/np.count_nonzero([reject[3],reject[10]]) - (ear_eeg[4]+ear_eeg[5]+ear_eeg[6]+ear_eeg[11])/np.count_nonzero([reject[4],reject[5],reject[6],reject[11]]) # (ERA + ERB)/2 - (ERE + ERI + ERG + ERK)/4
#     else:
#         R_E = np.zeros(data.shape[1])


#     return L_R, L_E, R_E

def EEG_process(data, rejected_list, i=0):
    # data 的形状是 (N_epochs, 20, 6000)
    # 我们需要沿通道维度（第 1 维）提取前 12 个耳电通道
    # 提取后，每个通道的形状变为 (N_epochs, 6000)
    ear_eeg = [data[:, j, :] for j in range(12)]

    reject = rejected_list[i]
    
    # 乘上拒绝掩码 (如果是 0 则该通道数据全为 0)
    for j in range(len(reject)):
        ear_eeg[j] = ear_eeg[j] * reject[j]

    # --- 安全除法计算 (防止全被 reject 导致除以 0) ---
    n_L = np.count_nonzero([reject[0], reject[1], reject[2], reject[7], reject[8], reject[9]])
    n_R = np.count_nonzero([reject[3], reject[4], reject[5], reject[6], reject[10], reject[11]])
    
    # 用 max(n, 1) 防止除零报错。如果全被 reject，分子本来就是 0，0/1=0，逻辑完美
    Left_ear = (ear_eeg[0] + ear_eeg[1] + ear_eeg[2] + ear_eeg[7] + ear_eeg[8] + ear_eeg[9]) / max(n_L, 1)
    Right_ear = (ear_eeg[3] + ear_eeg[4] + ear_eeg[5] + ear_eeg[6] + ear_eeg[10] + ear_eeg[11]) / max(n_R, 1)
    
    L_R = Left_ear - Right_ear

    # L_E 计算
    n_LE = np.count_nonzero([reject[0], reject[7]])
    n_LE_sub = np.count_nonzero([reject[1], reject[2], reject[8], reject[9]])
    if n_LE != 0:
        L_E = (ear_eeg[0] + ear_eeg[7]) / n_LE - (ear_eeg[1] + ear_eeg[2] + ear_eeg[8] + ear_eeg[9]) / max(n_LE_sub, 1)
    else:
        # 使用 zeros_like 自动匹配 (N_epochs, 6000) 形状
        L_E = np.zeros_like(Left_ear)

    # R_E 计算
    n_RE = np.count_nonzero([reject[3], reject[10]])
    n_RE_sub = np.count_nonzero([reject[4], reject[5], reject[6], reject[11]])
    if n_RE != 0:
        R_E = (ear_eeg[3] + ear_eeg[10]) / n_RE - (ear_eeg[4] + ear_eeg[5] + ear_eeg[6] + ear_eeg[11]) / max(n_RE_sub, 1)
    else:
        R_E = np.zeros_like(Right_ear)

    return L_R, L_E, R_E