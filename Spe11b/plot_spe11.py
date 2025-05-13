import sys
import numpy as np
import matplotlib.pyplot as plt
from input_output.get_ecl_key_val import *
from scipy.io import loadmat
import os


# Set paths and find results
path_to_files = '.'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)
seis_data = ['sim2seis', 'bulkimp']
non_scalar = seis_data + ['rft']


def plot_seis_2d(scaling=1.0, vintage=0):
    """
    Plot seismic 2D data (e.g. amplitude maps)

    Input:
        - scaling: if scaling of seismic data is used during data assimilation, this input can be used to convert back
                   to the original values
        - vintage: plot this vintage

    % Copyright (c) 2023 NORCE, All Rights Reserved.

    """

    wells = None
    if os.path.exists('wells.npz'):
        wells = np.load('wells.npz')['wells']

    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)
    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    obs_rec = None
    if os.path.exists('prior_forecast_rec.npz'):  # the amplitude map is the actual data
        obs_rec = np.load(str(path_to_files) + f'/truedata_rec_{vintage}.npz', allow_pickle=True)['arr_0']
        pred1 = np.load(str(path_to_files) + '/prior_forecast_rec.npz', allow_pickle=True)['arr_0']
        pred2 = np.load(str(path_to_files) + '/rec_results.p', allow_pickle=True)
    else:
        pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
        pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']

    # get the data
    data_obs = np.empty([])
    data1 = np.empty([])
    data2 = np.empty([])
    current_vint = 0
    for i, key in ((i, key) for _, i in enumerate(assim_index) for key in seis_data):
        if key in obs[i] and obs[i][key] is not None:
            if current_vint < vintage:
                current_vint += 1
                continue
            if type(pred2) is list:
                data1 = pred1[current_vint, :, :] / scaling
                data1 = data1.T
                data2 = pred2[current_vint] / scaling
                data_obs = obs_rec / scaling
            else:
                data1 = pred1[i][key] / scaling
                data2 = pred2[i][key] / scaling
                data_obs = obs[i][key] / scaling
            break

    # map to 2D
    if os.path.exists(f'mask_{vintage}.npz'):
        mask = np.load(f'mask_{vintage}.npz', allow_pickle=True)['mask']
    else:
        print('Mask is required to plot 2D data!')
        sys.exit()
    if os.path.exists('utm.mat'):
        sx = loadmat('utm.mat')['sx']
        sy = loadmat('utm.mat')['sy']
    else:
        sx = np.linspace(0, mask.shape[1], num=mask.shape[1])
        sy = np.linspace(mask.shape[0], 0, num=mask.shape[0])

    data = np.nan * np.ones(mask.shape)
    data[mask] = data_obs
    cl = np.array([np.min(data_obs), np.max(data_obs)])
    data1_mean = np.nan * np.ones(mask.shape)
    data1_mean[mask] = np.mean(data1, 1)
    data2_mean = np.nan * np.ones(mask.shape)
    data2_mean[mask] = np.mean(data2, 1)
    data1_std = np.nan * np.ones(mask.shape)
    data1_std[mask] = np.std(data1, 1)
    data2_std = np.nan * np.ones(mask.shape)
    data2_std[mask] = np.std(data2, 1)
    data1_min = np.nan * np.ones(mask.shape)
    data1_min[mask] = np.min(data1, 1)
    data2_min = np.nan * np.ones(mask.shape)
    data2_min[mask] = np.min(data2, 1)
    data1_max = np.nan * np.ones(mask.shape)
    data1_max[mask] = np.max(data1, 1)
    data2_max = np.nan * np.ones(mask.shape)
    data2_max[mask] = np.max(data2, 1)
    data_diff = data2_mean - data1_mean
    data_diff[np.abs(data_diff) < 0.01] = np.nan

    # compute the misfit
    v = data1_mean.flatten() - data.flatten()
    n = np.count_nonzero(~np.isnan(v))
    data1_misfit_mean = np.nansum(np.abs(v)) / n
    v = data2_mean.flatten() - data.flatten()
    n = np.count_nonzero(~np.isnan(v))
    data2_misfit_mean = np.nansum(np.abs(v)) / n
    data1_misfit_mean_str = str(data1_misfit_mean)
    data2_misfit_mean_str = str(data2_misfit_mean)
    reduction_str = str((data1_misfit_mean - data2_misfit_mean) * 100 / data1_misfit_mean)
    print('Initial misfit: ' + data1_misfit_mean_str)
    print('Final misfit  : ' + data2_misfit_mean_str)
    print('Reduction (%) : ' + reduction_str)

    colorm = 'viridis'
    plt.figure()
    im = plt.pcolormesh(sx, sy, data, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Data')
    filename = str(path_to_figures) + '/data_true' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_mean, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Initial simulated mean')
    filename = str(path_to_figures) + '/data_mean_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data2_mean, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Final simulated mean')
    filename = str(path_to_figures) + '/data_mean_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data_diff, cmap='seismic', shading='auto')
    cl_value = np.nanmax(np.abs(data_diff))
    cl_diff = np.array([-cl_value, cl_value])
    im.set_clim(cl_diff)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Final - Initial (trunc 0.01)')
    filename = str(path_to_figures) + '/data_mean_diff' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, data1_std, cmap=colorm, shading='auto')
    plt.colorbar()
    plt.title('Initial seismic std')
    filename = str(path_to_figures) + '/data_std_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, data2_std, cmap=colorm, shading='auto')
    plt.colorbar()
    plt.title('Final seismic std')
    filename = str(path_to_figures) + '/data_std_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_min, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Initial seismic min')
    filename = str(path_to_figures) + '/data_min_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    data2_min = data2_min
    im = plt.pcolormesh(sx, sy, data2_min, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Final seismic min')
    filename = str(path_to_figures) + '/data_min_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_max, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Initial seismic max')
    filename = str(path_to_figures) + '/data_max_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data2_max, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Final seismic max')
    filename = str(path_to_figures) + '/data_max_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')


def combined():
    """
    Plot objective function for all data combined

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    mm = []
    for iter in range(num_iter):
        if iter == 0:
            mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter + 1))['prev_data_misfit'])
        mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter + 1))['data_misfit'])

    f = plt.figure()
    plt.plot(mm, 'ko-')
    plt.xticks(np.arange(0, num_iter + 1), np.arange(num_iter + 1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Iteration no.', size=20)
    plt.ylabel('Data mismatch', size=20)
    plt.title('Objective function', size=20)
    f.tight_layout(pad=2.0)
    plt.savefig(str(path_to_figures) + '/obj_func')

plot_seis_2d(1.0, 4)
combined()

# Plot permx
dim = [83,58]
plt.figure()
true=read_file('PERMX', 'TRUE_PERMX.INC')
true=np.reshape(true, dim).T
plt.imshow(true)
plt.colorbar()
plt.title('true permx')
plt.savefig('Figures/true_permx')

plt.figure()
prior=np.load('prior.npz',allow_pickle=True)
prior=prior['permx']
priormean=np.mean(prior,1)
priormean=np.reshape(priormean,dim).T
plt.imshow(np.exp(priormean))
plt.colorbar()
plt.title('prior mean permx')
plt.savefig('Figures/prior_permx')

plt.figure()
post=np.load('posterior_state_estimate.npz',allow_pickle=True)
post = post['permx']
postmean=np.mean(post,1)
postmean = np.reshape(postmean, dim).T
plt.imshow(np.exp(postmean))
plt.colorbar()
plt.title('post mean permx')
plt.savefig('Figures/post_permx')

plt.show()