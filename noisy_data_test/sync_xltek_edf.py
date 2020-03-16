import matplotlib
matplotlib.use('TkAgg')

import wonambi.ioeeg as wio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path


sync_file = "/media/moneylab/VideoAnnotation/OpenPose/MG120/all/MG120_Xltek_Raw/Xxxxxxx~ Xxxxx_3d3e4428-1956-4519-88f5-ab24d00e8f16/sync/Xxxxxxx~ Xxxxx_3d3e4428-1956-4519-88f5-ab24d00e8f16_video_sync.csv"
sync = pd.read_csv(sync_file)
xltek_info = sync.loc[1]

xltek_file = "/media/moneylab/VideoAnnotation/OpenPose/MG120/all/MG120_Xltek_Raw/Xxxxxxx~ Xxxxx_3d3e4428-1956-4519-88f5-ab24d00e8f16/xltek_data/Xxxxxxx~ Xxxxx_3d3e4428-1956-4519-88f5-ab24d00e8f16_0001.npy"
xltek_data = np.load(xltek_file)
xltek_chan_data = xltek_data[1]
xltek_chan_data = xltek_chan_data - np.nanmean(xltek_chan_data)

edf_file = "/media/cashlab/SSD1/MG120/MG120_d10_Sun.edf"
edf_obj = wio.edf.Edf(edf_file)
edf_hdr = edf_obj.return_hdr()
edf_chan_data = edf_obj.return_dat([0], int(xltek_info['start sample'] - 5000), int(xltek_info['end sample'] + 5000))
edf_chan_data = edf_chan_data.reshape(-1)
edf_chan_data = edf_chan_data - np.nanmean(xltek_chan_data)
edf_chan_data = - edf_chan_data

corr = signal.correlate(edf_chan_data, xltek_chan_data)
shift = corr.argmax() - (len(xltek_chan_data) - 1)

plt.plot(xltek_chan_data, label='xltek')
plt.plot(edf_chan_data[shift:shift+len(xltek_chan_data)], label='edf')
plt.legend()

out_loc = Path("/media/jane/Projects/Volitional_Movement/AZ_projects/noisy_data_test")
np.save(out_loc / 'xltek_data', xltek_chan_data)
np.save(out_loc / 'edf_data', edf_chan_data[shift:shift+len(xltek_chan_data)])