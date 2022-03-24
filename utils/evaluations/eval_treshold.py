from cProfile import label
import glob
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def main(path):
    dirs = glob.glob(path)

    print(f"Number of results found {dirs.__len__()}")

    f1_corners = []
    f1_rooms = []
    labels = []
    for d in dirs:
        labels.append(float(d.split("_")[-1]))
        results = os.path.join(d, "360_dfpe_result.csv")
        assert os.path.isfile(results)
        data = pd.read_csv(results, header=None).values
        res = data[1:-1, 1:].astype(np.float)
        
        # ! F1-score for corners
        f1_cr = 2* res[:, 0] * res[:, 1]/(res[:, 0] + res[:, 1])
        f1_corners.append((np.nanmean(f1_cr), np.nanstd(f1_cr)))
        
        #! F1-score for rooms
        f1_r = 2* res[:, 6] * res[:, 7]/(res[:, 6] + res[:, 7])
        f1_rooms.append((np.nanmean(f1_r), np.nanstd(f1_r)))
    
    f1_corners = np.vstack(f1_corners)
    f1_rooms = np.vstack(f1_rooms)
    plt.figure("F1-score")
    plt.plot(labels, f1_corners[:, 0], label="F1-score Corners")
    plt.plot(labels, f1_rooms[:, 0], label="F1-score Room")
    plt.fill_between(labels, f1_corners[:, 0]-f1_corners[:, 1]**2, f1_corners[:, 0]+f1_corners[:, 1]**2,alpha =0.5)
    plt.fill_between(labels, f1_rooms[:, 0]-f1_rooms[:, 1]**2, f1_rooms[:, 0]+f1_rooms[:, 1]**2,alpha =0.5)
    plt.legend()
    plt.grid()
    plt.show()   
    print('done')


if __name__ == '__main__':
    results_path = "/HD/NFS/ycliu/360dfpe-reimp-dump/room_id_thrs*"
    main(results_path)
