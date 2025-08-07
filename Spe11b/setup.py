import datetime as dt
from misc import ecl
import csv
from subprocess import call,DEVNULL
from simulator.rockphysics.standardrp import elasticproperties
from copy import deepcopy
from input_output.get_ecl_key_val import *
from geostat.decomp import Cholesky  # Making realizations

# define the test case
case_name = 'SPE11B'
start = dt.datetime(2025, 1, 1)
dim = [83,58]

# define the data
seis_data = ['bulkimp']

def main():

    #  Generate prior ensemble
    _gen_prior()

    # Run file
    com = ['flow','--output-dir=TRUE_RUN', '--tolerance-mb=1e-7', '--linear-solver=cprw', '--enable-tuning=true', '--newton-min-iterations=1', '--enable-opm-rst-file=true', '--output-extra-convergence-info=steps,iterations', '--enable-well-operability-check=false', '--min-time-step-before-shutting-problematic-wells-in-days=1e-99', f'TRUE_RUN/{case_name}.DATA']
    call(com, stdout=DEVNULL)

    case = ecl.EclipseCase(f'TRUE_RUN/{case_name}')

    rpt = case.report_dates()
    report_time = [(el - start).days for el in rpt][1:]
    assim_time = report_time[1:]

    pem_input = {'vintage': report_time}

    abs_var = ['ABS', 100]

    f = open('true_data.csv', 'w', newline='')
    g = open('var.csv', 'w', newline='')
    h = open('true_data_index.csv','w',newline='')
    k = open('assim_index.csv','w',newline='')
    l = open('datatyp.csv','w',newline='')

    writer1 = csv.writer(f)
    writer2 = csv.writer(g)
    writer3 = csv.writer(h)
    writer4 = csv.writer(k)
    writer5 = csv.writer(l)


    for time in assim_time:
        tmp_data = []
        tmp_var = []
        list_datatyp = []

        if time in pem_input['vintage']:
            tmp_data.extend([f'{seis_data[0]}_{pem_input["vintage"].index(time)}.npz'])
            tmp_var.extend(abs_var)
        else:
            tmp_data.extend(['N/A'])
            tmp_var.extend(['REL','N/A'])
        list_datatyp.extend(seis_data)

        if time == assim_time[1]:
            for el in list_datatyp:
                writer5.writerow([el])
            writer3.writerow(assim_time)
            for i in range(len(assim_time)):
                writer4.writerow([i])
        writer1.writerow(tmp_data)
        writer2.writerow(tmp_var)


    f.close()
    g.close()
    h.close()
    l.close()
    k.close()

    # generate seismic data and noise
    np.savez('overburden.npz', **{'obvalues': 400. * np.ones(np.prod(dim))})

    elprop_input = {'overburden': 'overburden.npz',
                    'baseline': 0}
    elprop = elasticproperties(elprop_input)
    _pem(pem_input, case, elprop, start)


def _gen_prior():
    facies = read_file('SATNUM', 'SATNUM.INC')
    facies = np.reshape(facies, np.flip(dim)).flatten(order='F')
    N = 100
    param_log_mean = [-2.289, 4.618, 5.311, 6.228, 6.921, 7.614, -11.56]
    param_var = [0.01, 0.09, 0.09, 0.09, 0.09, 0.09, 0.01]
    corr_length = [1, 40, 40, 40, 40, 40, 1]
    aniso = [1, 4, 4, 4, 4, 4, 1]
    angle = [0, 45, 45, 45, 45, 45, 0]
    init_en = Cholesky()
    prior = {'permx': np.zeros((np.prod(dim), N))}
    true_perm = np.zeros((np.prod(dim)))
    for j in range(7):
        param_mean = param_log_mean[j] - param_var[j] / 2
        cov = init_en.gen_cov2d(dim[0], dim[1], param_var[j], corr_length[j], aniso[j], angle[j], 'sph')
        #param_field_facies = fast_gaussian(grid_dim, eval(f'{param}_info')['std'], corr)[act]
        ensemble = init_en.gen_real(np.ones(np.prod(dim)) * param_mean, cov, N)
        true_facies_perm = init_en.gen_real(np.ones(np.prod(dim)) * param_mean, cov, 1)
        true_perm[facies == j] = np.exp(true_facies_perm[facies == j, 0])
        prior['permx'][facies == j, :] = ensemble[facies == j, :]
        print('Generate permx for facies: ' + str(j+1))
    np.savez('prior.npz', **prior)
    write_file('TRUE_PERMX.INC', 'PERMX', true_perm)

def _pem(input, ecl_case, pem, startDate):
    grid = ecl_case.grid()
    phases = ecl_case.init.phases
    if 'WAT' in phases and 'GAS' in phases:  # This should be extended
        vintage = []
        # loop over seismic vintages
        for v,assim_time in enumerate(input['vintage']):
            time = startDate + \
                   dt.timedelta(days=assim_time)
            pem_input = {}
            # get active porosity
            tmp = ecl_case.cell_data('PORO')
            if 'compaction' in input:
                multfactor = ecl_case.cell_data('PORV_RC', time)

                pem_input['PORO'] = np.array(multfactor[~tmp.mask] * tmp[~tmp.mask], dtype=float)
            else:
                pem_input['PORO'] = np.array(tmp[~tmp.mask], dtype=float)
            # get active NTG if needed
            if 'ntg' in input:
                if input['ntg'] == 'no':
                    pem_input['NTG'] = None
                else:
                    tmp = ecl_case.cell_data('NTG')
                    pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)
            else:
                tmp = ecl_case.cell_data('NTG')
                pem_input['NTG'] = np.array(tmp[~tmp.mask], dtype=float)

            pem_input['RS'] = None
            for var in ['SWAT', 'SGAS', 'PRESSURE']:
                tmp = ecl_case.cell_data(var, time)
                pem_input[var] = np.array(tmp[~tmp.mask], dtype=float)  # only active, and conv. to float

            if 'press_conv' in input:
                pem_input['PRESSURE'] = pem_input['PRESSURE'] * input['press_conv']

            tmp = ecl_case.cell_data('PRESSURE', 1)
            if hasattr(pem, 'p_init'):
                P_init = pem.p_init * np.ones(tmp.shape)[~tmp.mask]
            else:
                P_init = np.array(tmp[~tmp.mask], dtype=float)  # initial pressure is first

            if 'press_conv' in input:
                P_init = P_init * input['press_conv']

            saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
                           for ph in phases]
            # Get the pressure
            pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
                                ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=P_init,
                                ensembleMember=None)

            tmp_value = np.zeros(ecl_case.init.shape)
            tmp_value[ecl_case.init.actnum] = pem.bulkimp

            pem.bulkimp = np.ma.array(data=tmp_value, dtype=float,
                                           mask=deepcopy(ecl_case.init.mask))
            # run filter
            pem._filter()
            vintage.append(deepcopy(pem.bulkimp))

        for i, elem in enumerate(vintage[1:]):
            pem_result = (elem - vintage[0])
            np.savez(f'bulkimp_{i+1}.npz',**{'bulkimp':pem_result})


if __name__ == '__main__':
    main()
