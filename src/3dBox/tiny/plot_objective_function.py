import numpy as np
import matplotlib.pyplot as plt
import os


# Set paths and find results
path_to_files = './SaveOutputs'
path_to_figures = './Figures'  # Save here
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)


def combined():
    """
    Plot objective function for all data combined
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    mm = []
    for iter in range(num_iter):
        if iter == 0:
            mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter + 1))['prev_data_misfit'])
        mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter+1))['data_misfit'])

    f = plt.figure()
    plt.plot(mm, 'ko-')
    plt.xticks(np.arange(0, num_iter+1), np.arange(num_iter+1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Iteration no.', size=20)
    plt.ylabel('Data mismatch', size=20)
    plt.title('Objective function', size=20)
    f.tight_layout(pad=2.0)
    plt.savefig(str(path_to_figures) + '/obj_func')
    plt.show()
    plt.close('all')


combined()

