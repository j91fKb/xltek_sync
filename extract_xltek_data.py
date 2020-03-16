"""
Extract xltek data for an entire subject's worth of clinical videos.
"""

import os
from pathlib import Path
from natsort import natsorted
import numpy as np
import pandas as pd
import wonambi.ioeeg.ktlx as kx


mad_dir = "/media/mad/Original Data/MG/MG120/MG120_Xltek_Raw"
mad_dir = Path(mad_dir)

main_dir = "/media/moneylab/VideoAnnotation/OpenPose/MG120/all/clinical"
main_dir = Path(main_dir)

xltek_dirs = natsorted([f for f in os.listdir(main_dir) if (main_dir / f).is_dir()])

for xltek_dir in xltek_dirs:
    print(xltek_dir)

    sync_file = main_dir / xltek_dir / 'sync' / (xltek_dir + '_video_sync.csv')
    if not sync_file.is_file():
        continue
    sync = pd.read_csv(sync_file)

    out_loc = main_dir / xltek_dir / 'xltek_data'
    if not out_loc.is_dir():
        os.mkdir(out_loc)

    xltek_obj = kx.Ktlx(mad_dir / xltek_dir)
    hdr = xltek_obj.return_hdr()
    channels = [i for i in range(len(hdr[3]))]

    for ind, video in sync.iterrows():
        print(ind)

        save_loc = out_loc / video['name'].replace('.avi', '.npy')
        if not save_loc.is_file() and video['notes'] != 'outside':
            data = xltek_obj.return_dat(channels, int(video['start sample']), int(video['end sample']))
            np.save(save_loc, data)
