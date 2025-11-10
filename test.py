import numpy as np
try:
    labels = np.load(r"C:\Users\28199\Desktop\dataset\PTBXL\Label\label.npy")
    print("标签文件加载成功，样本数:", len(labels))
except Exception as e:
    print("标签文件损坏:", e)