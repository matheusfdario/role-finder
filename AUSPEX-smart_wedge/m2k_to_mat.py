#!/usr/bin/env python

"""Script para execução da aplicação que converte dados de inspeção do formato M2K para o Matlab."""
import os
import sys
import argparse
import framework.file_m2k
import scipy.io as sio
import numpy as np

# Inclusão dos packages do projeto AUSPEX no PYTHONPATH.
sys.path.append(os.path.join(os.path.dirname(__file__), "framework"))
sys.path.append(os.path.join(os.path.dirname(__file__), "guiqt"))
sys.path.append(os.path.join(os.path.dirname(__file__), "imaging"))
sys.path.append(os.path.join(os.path.dirname(__file__), "surface"))
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))

if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("m2k_file", help="Caminho para arquivo M2K")
    ap.add_argument("mat_file", help="Caminho para arquivo MAT")
    ap.add_argument("-f", "--freq_transd", type=np.float32, default=5.0)
    ap.add_argument("-b", "--bw_transd", type=np.float32, default=0.5)
    ap.add_argument("-t", "--tp_transd", choices=['gaussian', 'cossquare', 'hanning', 'hamming'],
                    default='gaussian')
    ap.add_argument("-i", "--type_insp", choices=['contact', 'immersion'], default='contact')
    ap.add_argument("-w", "--water_path", type=np.float32, default=0.0)
    ap.add_argument("-s", "--sel_shots", nargs='+', type=int, default=None)
    args = vars(ap.parse_args())

    print('Loading M2K file ...')
    data_insp = framework.file_m2k.read(args['m2k_file'],
                                        args['freq_transd'],
                                        args['bw_transd'],
                                        args['tp_transd'],
                                        sel_shots=args['sel_shots'],
                                        water_path=args['water_path'],
                                        type_insp=args['type_insp']
                                        )

    if data_insp:
        if data_insp.inspection_params.water_path is None:
            data_insp.inspection_params.water_path = args['water_path']
        print('Generating MAT file ...')
        sio.savemat(args['mat_file'], {'ascans': data_insp.ascan_data,
                                       'inspection_params': data_insp.inspection_params,
                                       'probe_params': data_insp.probe_params,
                                       'specimen_params': data_insp.specimen_params,
                                       'time_grid': data_insp.time_grid})

        print('Done.')
    else:
        print('Error in M2K file processing.')
