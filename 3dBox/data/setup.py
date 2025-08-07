__author__ = 'kfo005'
import datetime as dt
import pickle

from misc import ecl,grdecl
import csv, os, shutil
import numpy as np
from mako.lookup import TemplateLookup
from mako.runtime import Context
from subprocess import call,Popen,PIPE,DEVNULL
from subsurface.rockphysics.standardrp import elasticproperties
from copy import deepcopy
from geostat import gaussian_sim
import mat73,shutil,glob


# define the test case
model = 'tiny'  # 'tiny', 'small', 'medium', 'large', or 'flowrock'
case_name = 'RUNFILE'
start = dt.datetime(2022, 1, 1)

# define the data
prod_wells = ['PRO1', 'PRO2', 'PRO3']
inj_wells = ['INJ1', 'INJ2', 'INJ3']
prod_data = ['WOPR', 'WWPR']
inj_data = ['WWIR']
seis_data = []
if model == 'flowrock':
  seis_data.append('bulkimp') # 'sim2seis' or 'bulkimp'

def main():
    # get information about the grid
    grid = grdecl.read(f'../{model}/grid/Grid.grdecl')

    np.random.seed(10) # fix the seed
    permx = 3.5*np.ones(np.product(grid['DIMENS'])) + gaussian_sim.fast_gaussian(grid['DIMENS'], np.array([1]), np.array([10, 10, 10]))
    #build the data file
    # Look for the mako file
    lkup = TemplateLookup(directories=os.getcwd(),
                          input_encoding='utf-8')
    tmpl = lkup.get_template(f'{case_name}.mako')
    if os.path.exists('TRUE_RUN'):
        shutil.rmtree('TRUE_RUN')
    os.mkdir('TRUE_RUN') # folder for run
    # use a context and render onto a file
    with open(f'TRUE_RUN/{case_name}.DATA','w') as f:
        ctx = Context(f, **{'model':model,'permx':permx})
        tmpl.render_context(ctx)

    # Run file
    com = ['flow','--output-dir=TRUE_RUN', f'TRUE_RUN/{case_name}.DATA']
    call(com, stdout=DEVNULL)

    case = ecl.EclipseCase(f'TRUE_RUN/{case_name}')

    rpt = case.report_dates()
    assim_time = [(el - start).days for el in rpt][1:]

    if len(seis_data) > 0:
        pem_input = {'vintage': [assim_time[4], assim_time[-1]]}
    #N=100
    #for i in range(len(pem_input['vintage'])):
    #    tmp_error = [gaussian_sim.fast_gaussian(np.array(list(dim_field)), np.array([800]),np.array([20])) for _ in range(N)]
    #    np.savez(f'var_bulk_imp_vintage_{i}.npz',error=np.array(tmp_error).T)

    rel_var = ['REL', 10]
    abs_var = {'WOPR':['ABS', 8**2],
               'WWPR':['ABS', 8**2],
               'WWIR':['ABS', 8**2],
               }


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
        for data in prod_data:
            for well in prod_wells:
                # same std for all data
                single_data = case.summary_data(data + ' ' + well, start + dt.timedelta(days=time))
                #all_data = [case.summary_data(data + ' ' + well, start + dt.timedelta(days=timeidx)) for timeidx in assim_time if case.summary_data(data + ' ' + well, start + dt.timedelta(days=timeidx)) > 0]
                list_datatyp.extend([data + ' ' + well])
                # if the data has value below 10 we must make the variance absolute!!
                if single_data > 0:
                    tmp_var.extend(abs_var[data])
                    tmp_data.extend(single_data)
                else:
                    tmp_var.extend(['ABS','100'])
                    tmp_data.extend(['0.0'])

        for data in inj_data:
            for well in inj_wells:
                single_data = case.summary_data(data + ' ' + well, start + dt.timedelta(days=time))
                #all_data = [case.summary_data(data + ' ' + well, start + dt.timedelta(days=timeidx)) for timeidx in assim_time if case.summary_data(data + ' ' + well, start + dt.timedelta(days=timeidx)) > 0]
                list_datatyp.extend([data + ' ' + well])
                # if the data has value 10 we must make the variance absolute!!
                if single_data > 0:
                    tmp_data.extend(single_data)
                    tmp_var.extend(abs_var[data])
                else:
                    tmp_var.extend(['ABS','100'])
                    tmp_data.extend(['0.0'])

        for data in seis_data:
            if time in pem_input['vintage']:
                tmp_data.extend([f'{data}_{pem_input["vintage"].index(time)}.npz'])
                tmp_var.extend(rel_var)
            else:
                tmp_data.extend(['N/A'])
                tmp_var.extend(['REL','N/A'])
            list_datatyp.extend([data])

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

    if len(seis_data) > 0:
        # generate seismic data and noise
        np.savez('overburden.npz', **{'obvalues': 320. * np.ones(np.product(grid['DIMENS']))})  # (10**(4)*9.81*depth)/(10**(5))})

        elprop_input = {'overburden': 'overburden.npz',
                        'baseline': 0}
        elprop = elasticproperties(elprop_input)
        _pem(pem_input, case, elprop, start)



def _pem(input, ecl_case, pem, startDate):
    grid = ecl_case.grid()
    phases = ecl_case.init.phases
    if 'OIL' in phases and 'WAT' in phases and 'GAS' in phases:  # This should be extended
        vintage = []
        # loop over seismic vintages
        for v,assim_time in enumerate([0] + input['vintage']):
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
            for var in ['SWAT', 'SGAS', 'PRESSURE', 'RS']:
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
            np.savez(f'bulkimp_{i}.npz',**{'bulkimp':pem_result})




if __name__ == '__main__':
    main()
