import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
from scipy.io import loadmat
import datetime as dt
import os


# Set paths and find results
path_to_files = './SaveOutputs'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)
seis_data = ['sim2seis', 'bulkimp']
non_scalar = seis_data + ['rft']


def plot_prod():
    """
    Plot all production data
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    obs = np.load('./obs_var.npz', allow_pickle=True)['obs']
    var = np.load('./obs_var.npz', allow_pickle=True)['var']
    data_dates = np.genfromtxt('../data/reportdates.csv', delimiter=',', dtype=str, encoding=None)
    data_dates = np.array([dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in data_dates])
    assim_index = np.genfromtxt('../data/assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)

    pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
    pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']
    ref_data = []
    if os.path.exists(str(path_to_files) + '/ref_data.p'):
        with open(str(path_to_files) + '/ref_data.p', 'rb') as f:
            ref_data = pickle.load(f)

    # Time_step
    tot_key = [el for el in obs[0].keys() if el not in non_scalar]
    x_days = [data_dates[i] for i in assim_index]
    ne = pred1[0][list(pred1[0].keys())[0]].shape[1]  # get the ensemble size from here

    for k in tot_key:

        # Find a well number
        n = tot_key.index(k)
        my_data = tot_key[n]
        print(my_data)
        t1, t2 = my_data.split()

        data_obs = []
        data_var = []
        data1 = []
        data2 = []
        ref = []
        for ind, i in enumerate(assim_index):
            data_obs.append(obs[i][my_data])
            data_var.append(var[i][my_data])
            data1.append(pred1[i][my_data])
            data2.append(pred2[i][my_data])
            if ref_data:
                if my_data in ref_data[ind].keys():
                    ref.append(ref_data[ind][my_data])
                else:
                    ref.append(None)

        n_d_obs = np.empty(0)
        n_d_var = np.empty(0)
        x_d = np.empty(0)
        n_d1 = np.empty((ne, 0))
        x_d1 = np.empty(0)
        n_d2 = np.empty((ne, 0))
        x_d2 = np.empty(0)
        n_d_ref = np.empty(0)
        x_d_ref = np.empty(0)
        for ind, i in enumerate(assim_index):
            if data_obs[ind] is not None:
                n_d_obs = np.append(n_d_obs, data_obs[ind])
                n_d_var = np.append(n_d_var, data_var[ind])
                x_d = np.append(x_d, x_days[ind])
            if ref_data and ref[ind] is not None:
                n_d_ref = np.append(n_d_ref, ref[ind])
                x_d_ref = np.append(x_d_ref, x_days[ind])
            if data_obs[ind] is not None:
                n_d1 = np.append(n_d1, data1[ind].transpose(), axis=1)
                x_d1 = np.append(x_d1, x_days[ind])
            if data_obs[ind] is not None:
                n_d2 = np.append(n_d2, data2[ind].transpose(), axis=1)
                x_d2 = np.append(x_d2, x_days[ind])

        f = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(x_d1, np.percentile(n_d1, 90, axis=0), 'k')
        plt.plot(x_d1, np.percentile(n_d1, 100, axis=0), ':k')
        plt.plot(x_d1, np.percentile(n_d1, 10, axis=0), 'k')
        plt.plot(x_d1, np.percentile(n_d1, 0, axis=0), ':k')
        # Plot observations with 2 std error bars
        p1 = ax1.errorbar(x_d, n_d_obs, yerr=2*np.sqrt(n_d_var), fmt='.r', capsize=5, elinewidth=2, markeredgewidth=2)
        p2 = None
        if ref_data:
            p2 = plt.plot(x_d_ref, n_d_ref, 'g')
        ax1.fill_between(x_d1, np.percentile(n_d1, 100, axis=0), np.percentile(n_d1, 0, axis=0), facecolor='lightgrey')
        ax1.fill_between(x_d1, np.percentile(n_d1, 90, axis=0), np.percentile(n_d1, 10, axis=0), facecolor='grey')
        p3 = ax1.fill(np.nan, np.nan, 'lightgrey')
        p4 = ax1.fill(np.nan, np.nan, 'grey')
        p5 = plt.plot(x_d1, np.mean(n_d1, axis=0), 'orange')
        if p2:
            ax1.legend([p1, (p2[0],), (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'ref', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 14}, bbox_to_anchor=(1, -0.5), ncol=2)
        else:
            ax1.legend([p1, (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 14}, bbox_to_anchor=(1, -0.5), ncol=2)
        plt.title(str(t1) + ' initial forcast, at Well: ' + str(t2), size=20)
        ylim = plt.gca().get_ylim()
        ax1.set_ylim(ylim)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Dates', size=20)
        if "WBHP" in my_data:
            plt.ylabel('Bar', size=20)
        else:
            plt.ylabel('Sm3/Day', size=20)

        ax2 = plt.subplot(2, 1, 2)
        plt.plot(x_d2, np.percentile(n_d2, 90, axis=0), 'k')
        plt.plot(x_d2, np.percentile(n_d2, 100, axis=0), ':k')
        plt.plot(x_d2, np.percentile(n_d2, 10, axis=0), 'k')
        plt.plot(x_d2, np.percentile(n_d2, 0, axis=0), ':k')
        # Plot observations with 2 std error bars
        ax2.errorbar(x_d, n_d_obs, yerr=2*np.sqrt(n_d_var), fmt='.r', capsize=5, elinewidth=2, markeredgewidth=2)
        if ref_data:
            plt.plot(x_d_ref, n_d_ref, 'g')
        ax2.fill_between(x_d2, np.percentile(n_d2, 100, axis=0), np.percentile(n_d2, 0, axis=0), facecolor='lightgrey')
        ax2.fill_between(x_d2, np.percentile(n_d2, 90, axis=0), np.percentile(n_d2, 10, axis=0), facecolor='grey')
        plt.plot(x_d2, np.mean(n_d2, axis=0), 'orange')
        plt.title(str(t1) + ' final forcast, at Well: ' + str(t2), size=20)
        f.tight_layout(pad=1.0)
        plt.xlabel('Dates', size=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if "WBHP" in my_data:
            plt.ylabel('Bar', size=20)
        else:
            plt.ylabel('Sm3/Day', size=20)
        if save_figure is True:
            plt.savefig(str(path_to_figures) + '/' + str(t2) + '_' + str(t1) + '.png', format='png')

    ############
    plt.show()
    plt.close('all')


plot_prod()