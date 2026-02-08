__author__ = 'kfo005'
import datetime as dt
import pickle
import sys

from resdata.summary import Summary
from resdata.resfile import ResdataRestartFile,ResdataInitFile
from resdata.grid import Grid

from misc import ecl,grdecl

import pandas as pd


#from misc import ecl,grdecl
import csv, os, shutil
import numpy as np
from mako.lookup import TemplateLookup
from mako.runtime import Context
from subprocess import call,Popen,PIPE,DEVNULL

from subsurface.rockphysics.standardrp import elasticproperties

from copy import deepcopy
from geostat import gaussian_sim
import mat73,shutil,glob


# This scipt generates the true data for the ML data-assimilation study.

# NOTE: I use this to test how true data can be stored as a dataframe to simplify PET interaction. All data is extracted
# via the RESDATA package.

# define the test case
model = 'tiny'  # 'tiny', 'small', 'medium', 'large', or 'flowrock'
case_name = 'RUNFILE'

# define the data
prod_wells = ['PRO1', 'PRO2', 'PRO3']
inj_wells = ['INJ1']
prod_data = ['WOPR', 'WWPR']
inj_data = ['WWIR']
seis_data = ['bulkimp']

def main():

    grid = grdecl.read(f'../{model}/grid/Grid.grdecl')

    np.random.seed(10) # fix the seed
    permx = 3.5*np.ones(np.prod(grid['DIMENS'])) + gaussian_sim.fast_gaussian(grid['DIMENS'], np.array([1]), np.array([10, 10, 10])).flatten()
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
    
    case = Summary(f'TRUE_RUN/{case_name}')

    rpt = case.report_dates

    #assim_time = [(el - start).days for el in rpt][1:]

    #if len(seis_data) > 0:
    #    pem_input = {'vintage': [assim_time[4], assim_time[-1]]}
    #N=100
    #for i in range(len(pem_input['vintage'])):
    #    tmp_error = [gaussian_sim.fast_gaussian(np.array(list(dim_field)), np.array([800]),np.array([20])) for _ in range(N)]
    #    np.savez(f'var_bulk_imp_vintage_{i}.npz',error=np.array(tmp_error).T)

    rel_var = ['REL', 10]
    abs_var = {'WOPR':['ABS', 8**2],
               'WWPR':['ABS', 8**2],
               'WWIR':['ABS', 8**2],
               }

    ##################################################################
    ##################################################################
    # build a pandas dataframe with the data.
    # The tvd is the index and the tuple (freq,dist) is the columns

    data_keys = []
    for w in prod_wells:
        for data_typ in prod_data:
            data_key = f'{data_typ}:{w}'
            data_keys.append(data_key)
    for w in inj_wells:
        for data_typ in inj_data:
            data_key = f'{data_typ}:{w}'
            data_keys.append(data_key)


    # Calculate the bulk impedance at the restart times (the predefined vintages)
    # restart = ResdataRestartFile(Grid(f"TRUE_RUN/{case_name}"),f"TRUE_RUN/{case_name}.UNRST")
    # init = ResdataInitFile(Grid(f"TRUE_RUN/{case_name}"),f"TRUE_RUN/{case_name}.INIT")
    # vintages = restart.report_dates
    # bulkimp = []
    # pem_input = {'PORO': init["PORO"][0].numpy_copy(),
    #              'NTG': init["NTG"][0].numpy_copy(),
    #              "P_INIT": restart.restart_get_kw("PRESSURE",vintages[0]).numpy_copy()*0.1 # Bar to MPa
    #              }

    # # calculate the hydrostatic overburden pressure based on the depth
    # depth = init["DEPTH"][0].numpy_copy()

    # dens_rock = 2400 # kg/m3
    # overburden = depth * dens_rock * 9.81 / 1e6 # Pa to MPa
    # np.savez('overburden.npz', **{'obvalues': overburden})

    # pem = elasticproperties({'overburden': 'overburden.npz',
    #            'baseline': 0,
    #            'parallel':1})

    # phases = ['OIL', 'WAT', 'GAS']

    # for time in vintages:
    #     pem_input['PRESSURE'] = restart.restart_get_kw("PRESSURE", time).numpy_copy()*0.1 # Bar to MPa
    #     pem_input['SWAT'] = restart.restart_get_kw("SWAT", time).numpy_copy()
    #     try:
    #         pem_input['SGAS'] = restart.restart_get_kw("SGAS", time).numpy_copy()
    #     except:
    #         pem_input['SGAS'] = np.zeros_like(pem_input['SWAT'])

    #     try:
    #         pem_input['RS'] = restart.restart_get_kw("RS", time).numpy_copy()
    #     except:
    #         pem_input['RS'] = np.zeros_like(pem_input['SWAT'])

    #     saturations = [1 - (pem_input['SWAT'] + pem_input['SGAS']) if ph == 'OIL' else pem_input['S{}'.format(ph)]
    #                    for ph in phases]


    #     # Get the pressure
    #     pem.calc_props(phases, saturations, pem_input['PRESSURE'], pem_input['PORO'],
    #                    ntg=pem_input['NTG'], Rs=pem_input['RS'], press_init=pem_input['P_INIT'],
    #                    ensembleMember=None)

    #     bulkimp.append(deepcopy(pem.bulkimp))

    # # each data is the 4D difference between the bulkimp at the vintage and the first vintage
    # # make a df with the data.
    # difference_data = [bk - bulkimp[0] for bk in bulkimp[1:]]
    # bulkimp_df = pd.DataFrame({'bulkimp':difference_data},index=vintages[1:])

    data_df = case.pandas_frame(time_index=rpt,column_keys=data_keys)
    # data_df['bulkimp'] = bulkimp_df['bulkimp'].reindex(data_df.index).apply(
    #                    lambda x: x if isinstance(x, np.ndarray) else None
    #                     )
    
    # Replace ':' with ' ' in column headers
    data_df.columns = data_df.columns.str.replace(':', ' ')

    data_df.index.name = 'dates'
    #data_df.to_csv('data.csv', index=True)
    data_df.to_pickle('data.pkl')

    # Assume data_df is the merged dataframe, including a 'bulkimp' column of lists

    # Separate scalar columns
    scalar_cols = data_df.select_dtypes(include=[np.number]).columns
    percentile_1 = data_df[scalar_cols].quantile(0.01)

    min_std_value = 0.005

    # Compute std_values only for scalar columns
    std_values = data_df[scalar_cols].apply(
        lambda col: np.maximum(
            np.where(
                col < percentile_1[col.name],
                0.1 * percentile_1[col.name],
                0.1 * col
            ),
            min_std_value
        ),
        axis=0
    )

    # Transform scalar values into ["abs", val**2]
    var_scalar_df = std_values.map(lambda val: ["abs", val ** 2])

    # Step 2: Handle 'bulkimp' column
    # First compute 1st percentile across all list elements
    # flat_bulkimp = [x for sublist in data_df['bulkimp'].dropna() for x in sublist]
    # bulkimp_percentile_1 = np.percentile(flat_bulkimp, 1)

    # def transform_bulkimp(vec):
    #     if not isinstance(vec, np.ndarray):
    #         return None
    #     result = []
    #     for val in vec:
    #         std = max(0.005, max(0.1 * val, 0.1 * bulkimp_percentile_1))
    #         result.append(std ** 2)
    #     return ["abs"] + [result]

    # var_bulkimp = data_df['bulkimp'].apply(transform_bulkimp)

    # Step 3: Combine
    #var_df = pd.concat([var_scalar_df, var_bulkimp.rename('bulkimp')], axis=1)
    var_df = var_scalar_df

    var_df.index.name = 'dates'
    # Write the variance for each data point

    var_df.to_csv('var.csv', index=True)
    with open('var.pkl', 'wb') as f:
        pickle.dump(var_df, f)

    with open("datatyp.csv", "w") as f:
        f.write(",".join(data_df.columns))

    with open("assim_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for c, idx in enumerate(data_df.index):
            writer.writerow([str(c)])

    with open("reportdates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for c, idx in enumerate(data_df.index):
            writer.writerow([idx.strftime('%Y-%m-%d %H:%M:%S')])

    sys.exit()
    ######################

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
