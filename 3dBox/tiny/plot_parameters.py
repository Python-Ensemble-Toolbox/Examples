import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
from scipy.stats import norm
from misc import ecl, grdecl
from scipy.io import loadmat


# Set paths and find results
path_to_files = './SaveOutputs'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)


def plot_layer(field, f_dim, iter=1, layer_no=1):
    """
    Plot parameters in given layer

    Input:
        - field   : string specifying the property
        - f_dim   : dimension of the property (2d or 3d) (nz, ny, nx)
        - iter    : plot results at this iteration
        - layer_no: plot for this layer

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    if os.path.exists(str(path_to_files) + '/actnum.npz'):
        actnum = np.load(str(path_to_files) + '/actnum.npz')['actnum']
    else: 
        actnum = np.ones(np.prod(f_dim), dtype=bool)

    # Load debug steps
    field_post = np.zeros(f_dim)
    field_post[:] = np.nan
    field_post_std = np.zeros(f_dim)
    field_post_std[:] = np.nan
    post = np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)['state'][()][field]
    if 'perm' in field:
        post = np.exp(post)
    field_post[actnum.reshape(f_dim)] = post.mean(1)
    field_post_layer = field_post[layer_no - 1, :, :]
    field_post_std[actnum.reshape(f_dim)] = post.std(axis=1, ddof=1)
    field_post_std_layer = field_post_std[layer_no - 1, :, :]
    max_post_std = np.nanmax(field_post_std_layer)
    min_post_std = np.nanmin(field_post_std_layer)

    # Load Prior field
    prior = np.load('./prior_ensemble.npz')[field]
    field_prior = np.zeros(f_dim)
    field_prior_std = np.zeros(f_dim)
    field_prior[:] = np.nan
    field_prior_std[:] = np.nan
    if 'perm' in field:
        prior = np.exp(prior)
    field_prior[actnum.reshape(f_dim)] = prior.mean(1)
    field_prior_layer = field_prior[layer_no - 1, :, :]
    field_prior_std[actnum.reshape(f_dim)] = prior.std(axis=1, ddof=1)
    field_prior_std_layer = field_prior_std[layer_no - 1, :, :]
    max_prior_std = np.nanmax(field_prior_std_layer)
    min_prior_std = np.nanmin(field_prior_std_layer)

    # Load the true field
    ecl_true = ecl.EclipseCase('../data/TRUE_RUN/RUNFILE')
    true_perm = ecl_true.cell_data('PERMX')
    true_perm = true_perm[layer_no-1, :, :]

    # Plotting
    if os.path.exists('utm_res.mat'):
        sx = loadmat('utm_res.mat')['sx_res']
        sy = loadmat('utm_res.mat')['sy_res']
    else:
        sx = np.linspace(0, f_dim[1], num=f_dim[1])
        sy = np.linspace(0, f_dim[2], num=f_dim[2])

    # Load wells if present
    wells = None
    if os.path.exists('wells.npz'):
        wells = np.load('wells.npz')['wells']

    # true field
    plt.figure()
    plt.pcolormesh(sx, sy, true_perm, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'True, ' + field
    filename = str(path_to_figures) + '/' + field + '_true'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, field_prior_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Prior, ' + field
    filename = str(path_to_figures) + '/' + field + '_prior'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, field_post_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior, ' + field
    filename = str(path_to_figures) + '/' + field + '_post'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    field_diff = field_post_layer - field_prior_layer
    plt.pcolormesh(sx, sy, field_diff, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior - Prior, ' + field
    filename = str(path_to_figures) + '/' + field + '_diff'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    # std
    np.array([np.minimum(min_prior_std, min_post_std), np.maximum(max_prior_std, max_post_std)])
    plt.figure()
    plt.pcolormesh(sx, sy, field_prior_std_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Prior std ' + field
    filename = str(path_to_figures) + '/' + field + '_std_prior'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, field_post_std_layer, cmap='jet', shading='auto')
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    title_str = 'Posterior std ' + field
    filename = str(path_to_figures) + '/' + field + '_std_post'
    if f_dim[0] > 1:  # 3D
        title_str += ' at layer ' + str(layer_no)
        filename += '_layer' + str(layer_no)
    plt.title(title_str, size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if save_figure is True:
        plt.savefig(filename)
        os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.show()


def export_to_grid(propname):
    """
    Export a property to .grdecl file (for inspection in e.g., ResInsight)

    Input:
        - propname: string spesifying property name

    % Copyright (c) 2023 NORCE, All Rights Reserved.

    """

    # Load posterior property
    post = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz',
                   allow_pickle=True)['state'][()][propname]
    if 'perm' in propname:
        post = np.exp(post)

    # Load prior property
    prior = np.load('./prior_ensemble.npz')[propname]
    if 'perm' in propname:
        prior = np.exp(prior)

    # Active gridcells
    if os.path.exists(str(path_to_files) + '/actnum.npz'):
        actnum = np.load(str(path_to_files) + '/actnum.npz')['actnum']
    else:
        actnum = np.ones(prior.shape[0], dtype=bool)

    # Make the property on full grid
    field_post = np.zeros(actnum.shape)
    field_prior = np.zeros(actnum.shape)
    field_post[actnum] = post.mean(1)
    field_prior[actnum] = prior.mean(1)
    dim = len(actnum)

    trans_dict = {}

    def _lookup(kw):
        return trans_dict[kw] if kw in trans_dict else kw

    # Write a quantity to the grid as a grdecl file
    with open(path_to_files + '/prior_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'prior_'+propname, field_prior, _lookup, dim)
    with open(path_to_files + '/posterior_' + propname + '.grdecl', 'wb') as fileobj:
        grdecl._write_kw(fileobj, 'posterior_'+propname, field_post, _lookup, dim)


export_to_grid('permx')
plot_layer('permx', [2, 10, 10])
