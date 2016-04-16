#!/usr/bin/env python

import os
from glob import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from scipy import stats

from pisa.utils.jsons import from_json, to_json

columns = [
    'Geometry', 'Ordering', 'Method', 'theta23_injected', 'Livetime',
    'Significance', 'Error', 'Trials'
]
if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Extract Asimov results from all listed files and construct
        Pandas DataFrame with these''',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--outfile', required=True,
        help='Output Pandas HDF5 store.'
    )
    parser.add_argument(
        'files', nargs='+',
        help='All files you want results extracted from.'
    )
    args = parser.parse_args()

    metric = 'chisquare'

    all_data = []
    for infile_name in args.files:
        base = os.path.basename(infile_name)
        #geom = base[0:3].upper()
        data = from_json(infile_name)
        ts_p_vals = {p: sd['value'] for p, sd in data['template_settings']['params'].iteritems()}
        results = data['results']
        geom = ts_p_vals['aeff_slice_smooth'][11:14].upper()
        assert geom in ['V36', 'V38', 'V39', 'V40'], geom
        livetime = ts_p_vals['livetime']
        for ordering in ['normal', 'inverted']:
            if ordering == 'normal':
                t23 = ts_p_vals['theta23_nh']
                dm31 = ts_p_vals['deltam31_nh']
                delta_chisquare = (results['data_NMH']['hypo_IMH'][metric][-1] -
                                   results['data_NMH']['hypo_NMH'][metric][-1])
            else:
                t23 = ts_p_vals['theta23_ih']
                dm31 = ts_p_vals['deltam31_ih']
                delta_chisquare = (results['data_IMH']['hypo_NMH'][metric][-1] -
                                   results['data_IMH']['hypo_IMH'][metric][-1])

            t23 = np.rad2deg(t23)

            sig_1sdd = np.sqrt(delta_chisquare)
            p_1sdd = stats.norm.sf(sig_1sdd)
            p_2sdd = p_1sdd / 2.0
            sig_2sdd = stats.norm.isf(p_2sdd)

            all_data.append(dict(
                Geometry=geom,
                Ordering=ordering,
                Method='Asimov',
                theta23_injected=t23,
                Livetime=livetime,
                Significance=sig_2sdd,
                Error=np.nan,
                Trials=np.nan
            ))
    df = pd.DataFrame(data=all_data, columns=columns)
    df.to_hdf(args.outfile, key='results', mode='w', format='table')
