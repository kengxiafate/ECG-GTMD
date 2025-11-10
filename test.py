import numpy as np
try:
    labels = np.load(r"C:\Users\28199\Desktop\dataset\PTBXL\Label\label.npy")
    print("The label file has been loaded successfully. The number of samples:", len(labels))
except Exception as e:

    print("Tag file is damaged:", e)
