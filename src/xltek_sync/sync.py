import os
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wonambi.ioeeg.ktlx as kx
from pathlib import Path
from datetime import datetime, timedelta
from natsort import natsorted
from pandas.plotting import register_matplotlib_converters
from .utils import time_to_sample, timezone_shift

matplotlib.use('Agg')
register_matplotlib_converters()


class Sync:
    def __init__(self, xltek_dir, output_dir=None):
        # xltek folder
        self.read_loc = Path(xltek_dir)

        # output folder
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.read_loc

        # grab header information
        print('GRABBING INFORMATION...')
        try:
            self.hdr = kx.Ktlx(self.read_loc).return_hdr()
        except:
            # sometimes grabbing header information may fail
            raise Exception('Failed to grab header information.')
        self.channels = self._get_channels()

        # video data is a pandas data frame, includes sync information once complete
        self.vid = self._get_vid()
        self.snc = self._get_snc()  # sync trigger data is a pandas data frame

        # intialize other variables
        self.vid_frame_sample = pd.DataFrame([])

    def run(self):
        # sync it up
        self.sync()

        # get sample number for each frame
        self.vid_frame_sample = self._frame_to_sample()

        # save data
        self.save_data()

        # save plots
        self.save_plots()

    def _get_channels(self):
        channels = pd.DataFrame(data=self.hdr[3], columns=['channel'])
        return channels

    def _get_vid(self):
        hdr = self.hdr

        # set up a data frame for the video data
        vid_hdr = ['name', 'start time', 'end time', 'n frames',
                   'start sample', 'end sample', 'start frame', 'end frame',
                   'notes']
        vid = pd.DataFrame(np.nan, index=range(
            len(hdr[5]['vtc'][0])), columns=vid_hdr)

        # read video data from header and add to data frame
        # video data is in the format = file name, date start time, date end time
        vid.iloc[:, :3] = np.array(hdr[5]['vtc']).T

        # find the number of frames for each video
        for ind, vid_file in enumerate(vid['name']):
            vid_file = os.path.join(self.read_loc, vid_file)
            v = cv2.VideoCapture(vid_file)
            vid.loc[ind, 'n frames'] = v.get(cv2.CAP_PROP_FRAME_COUNT)

        return vid

    def _get_snc(self):
        hdr = self.hdr

        # set up a data frame for the sync data
        snc_hdr = ['sample', 'time', 'frame rate']
        snc = pd.DataFrame(np.nan, index=range(
            len(hdr[5]['snc'][0]) + 1), columns=snc_hdr)

        # read sync data from header and add to data frame
        # sync data is in the format = sample number, date time for this sample
        snc.iloc[1:, :2] = np.array(hdr[5]['snc']).T

        # frame rate seems to vary between sync windows, so we must calculate frame rate for each
        snc.loc[1:len(snc)-2, 'frame rate'] = (np.array(snc['sample'][2:]) - np.array(snc['sample'][1:-1])) \
            / np.array(list(map(timedelta.total_seconds, np.array(snc['time'][2:]) - np.array(snc['time'][1:-1]))))

        # estimate that the overall frame rate = initial epoch's frame rate (for sample 0 to first trigger)
        snc.loc[0, 'frame rate'] = (snc['sample'].iloc[-1] - snc['sample'][1]) \
            / (snc['time'].iloc[-1] - snc['time'][1]).total_seconds()

        # find system start time by using the estimated frame rate
        snc.loc[0, 'sample'] = 0
        snc.loc[0, 'time'] = kx.convert_sample_to_video_time(
            0, snc['frame rate'][0], list(
                snc['sample'][1:]), list(snc['time'][1:])
        )

        # correct for timezone shift between video and sync
        snc['time'] += timezone_shift(self.vid['start time']
                                      [0], snc['time'][0])

        return snc

    def sync(self):
        print('SYNCING XLTEK...')

        vid = self.vid
        snc = self.snc

        # initialize frame windows, not used in any calculations
        vid['start frame'] = 1
        vid['end frame'] = vid['n frames']

        # initialize counters for main loop
        snc_ind = 0
        vid_ind = 0

        # main loop
        while vid_ind < len(vid):

            # CASE 1
            # video start and end times are between current sync and next sync times
            if (snc_ind < len(snc) - 1) and \
                    (snc.loc[snc_ind, 'time'] <= vid['start time'][vid_ind] < snc['time'][snc_ind + 1]) and \
                    (vid['end time'][vid_ind] < snc['time'][snc_ind + 1]):

                # set video start sample with current sync time
                vid.loc[vid_ind, 'start sample'] = time_to_sample(
                    vid['start time'][vid_ind], snc['time'][snc_ind], snc['sample'][snc_ind], snc['frame rate'][snc_ind]
                )

                # set video end sample with current sync time
                vid.loc[vid_ind, 'end sample'] = time_to_sample(
                    vid['end time'][vid_ind], snc['time'][snc_ind], snc['sample'][snc_ind], snc['frame rate'][snc_ind]
                )

            # CASE 2 & 3
            # video start time is between current sync and next sync times and video end time is after next sync time
            elif (snc_ind < len(snc) - 1) and \
                    (snc['time'][snc_ind] <= vid['start time'][vid_ind] < snc['time'][snc_ind + 1]) and \
                    (vid['end time'][vid_ind] >= snc['time'][snc_ind + 1]):

                # set video start sample with current sync time
                vid.loc[vid_ind, 'start sample'] = time_to_sample(
                    vid['start time'][vid_ind], snc['time'][snc_ind], snc['sample'][snc_ind], snc['frame rate'][snc_ind]
                )

                # increment sync index until video end time is within 'current' sync time
                while (snc_ind < len(snc) - 1) and \
                        (vid['end time'][vid_ind] >= snc['time'][snc_ind + 1]):
                    snc_ind += 1

                # CASE 2
                # if video end time is before last sync time
                if snc_ind < len(snc) - 1:
                    # set video end sample with now updated sync time
                    vid.loc[vid_ind, 'end sample'] = time_to_sample(
                        vid['end time'][vid_ind], snc['time'][snc_ind], snc['sample'][snc_ind],
                        snc['frame rate'][snc_ind]
                    )

                # CASE 3
                # if video end time is after the last sync time
                else:
                    # set video end sample as last sync sample
                    vid.loc[vid_ind, 'end sample'] = snc['sample'].iloc[-1]

                    # update the end frame of the video to when sampling ends
                    vid.loc[vid_ind, 'end frame'] = round(
                        # length of video within sync times
                        (snc['time'].iloc[-1] - vid['start time'][vid_ind])
                        # total length of video
                        / (vid['end time'][vid_ind] - vid['start time'][vid_ind])
                        * vid['n frames'][vid_ind]  # number of frames
                    )
                    vid.loc[vid_ind, 'notes'] = 'overhang end'

            # CASE 4
            # video start time is after next sync time
            elif (snc_ind < len(snc) - 1) and \
                    (vid['start time'][vid_ind] >= snc['time'][snc_ind + 1]):

                # increment sync index until video start time is within a sync window
                while (snc_ind < len(snc) - 1) and \
                        (vid['start time'][vid_ind] >= snc['time'][snc_ind + 1]):
                    snc_ind += 1

                continue

            # CASE 5
            # video starts before first sync time and video ends after first sync time
            elif (snc_ind == 0) and \
                    (vid['start time'][vid_ind] < snc['time'][0]) and \
                    (vid['end time'][vid_ind] >= snc['time'][0]):

                # set video start sample as first sync sample
                vid.loc[vid_ind, 'start sample'] = snc['sample'][0]

                # increment sync index until video end time is within a sync window
                while (snc_ind < len(snc) - 1) and \
                        (vid['end time'][vid_ind] >= snc['time'][snc_ind + 1]):
                    snc_ind += 1

                # set video end sample
                vid.loc[vid_ind, 'end sample'] = time_to_sample(
                    vid['end time'][vid_ind], snc['time'][snc_ind], snc['sample'][snc_ind], snc['frame rate'][snc_ind]
                )

                # update the start frame of the video to when sampling starts
                vid.loc[vid_ind, 'start frame'] = round(
                    (snc['time'][0] - vid['start time'][vid_ind])
                    / (vid['end time'][vid_ind] - vid['start time'][vid_ind])
                    * vid['end frame'][vid_ind]
                )
                vid.loc[vid_ind, 'notes'] = 'overhang start'

            # CASE 6
            # video start and end times are both before first sync time or both after last sync time
            else:
                vid.iloc[vid_ind, 4:8] = np.nan
                vid.loc[vid_ind, 'notes'] = 'outside'

            # increment video index
            vid_ind += 1

        print("FINISHED SYNCING")

    def _frame_to_sample(self):
        vid = self.vid

        # set up a data frame
        # make enough rows for largest video
        hdr_row = np.arange(1, np.nanmax(vid['n frames'])+1)
        hdr_col = vid['name']  # column headers are video names
        vid_frame_sample = pd.DataFrame(np.nan, index=hdr_row, columns=hdr_col)
        vid_frame_sample.index.names = ['frame']

        # convert frames to samples
        for vid_ind, vid_name in enumerate(vid['name']):
            if not np.isnan(vid['start frame'][vid_ind]):
                # convert each frame to a fraction of the video duration
                vid_fraction = np.arange(vid['start frame'][vid_ind], vid['end frame'][vid_ind] + 1) \
                    / (vid['end frame'][vid_ind]) - (vid['start frame'][vid_ind] / vid['end frame'][vid_ind])

                # get the number of samples in the video
                vid_sample_length = vid['end sample'][vid_ind] - \
                    vid['start sample'][vid_ind]

                # multiply video fraction with sample length and add to the starting sample
                vid_frame_sample[vid_name].loc[vid['start frame'][vid_ind]:vid['end frame'][vid_ind]] = np.round(
                    vid_fraction * vid_sample_length + vid['start sample'][vid_ind])

        return vid_frame_sample

    def _output_pathing(self):
        # get output
        output_dir = self.output_dir

        # make a subdirectory to store output
        folder_stem = output_dir / 'sync'
        if not os.path.isdir(folder_stem):
            os.mkdir(folder_stem)

        # filename stem
        file_stem = os.path.split(self.read_loc)[-1]

        return folder_stem, file_stem

    def save_data(self):
        folder_stem, file_stem = self._output_pathing()

        # save all data
        np.save(folder_stem / (file_stem + '_header.npy'), self.hdr)
        self.channels.to_csv(
            folder_stem / (file_stem + '_channels.csv'), index=False)
        self.vid.to_csv(folder_stem / (file_stem +
                                       '_video_sync.csv'), index=False)
        self.snc.to_csv(folder_stem / (file_stem +
                                       '_sync_triggers.csv'), index=False)
        self.vid_frame_sample.to_csv(
            folder_stem / (file_stem + '_video_frame_sample.csv'))

        print("SAVED DATA")

    def save_plots(self):
        plt.rcParams['figure.figsize'] = (18, 8)

        snc = self.snc
        vid = self.vid

        folder_stem, file_stem = self._output_pathing()

        # plot sync
        plt.title('Sync (should look linear and aligned)')
        plt.xlabel('Date Time')
        plt.ylabel('Sample')
        plt.plot(np.array(vid[['start time', 'end time']]).reshape(-1),
                 np.array(vid[['start sample', 'end sample']]).reshape(-1),
                 'k.', markersize=2, label='Video')
        plt.plot(snc['time'], snc['sample'], 'b|',
                 markersize=10, label='Sync Triggers')
        plt.legend()
        plt.savefig(folder_stem / (file_stem + '_sync.png'))
        plt.close()

        # plot sampling frequency
        plt.title('Sampling Frequency (should be consistent)')
        plt.xlabel('Date Time')
        plt.ylabel('Sampling Frequency (Hz)')
        x = np.ndarray.flatten(
            np.array([snc['time'].iloc[:-1], snc['time'].iloc[1:]]), order='F')
        y = np.ndarray.flatten(
            np.array([snc['frame rate'], snc['frame rate']]), order='F')[:-2]
        plt.plot(x, y, 'k')

        # adjust sampling frequency y axis if fluctuations are small
        thresh = .002*snc['frame rate'][0]
        if np.nanmax(np.abs(snc['frame rate'] - snc['frame rate'][0])) < thresh:
            plt.ylim(round(snc['frame rate'][0]-thresh),
                     round(snc['frame rate'][0]+thresh))
        plt.savefig(folder_stem / (file_stem + '_frame_rate.png'))
        plt.close()

        print('SAVED PLOTS')

# WIP
#
# # check erd files
# stc_entries = xltek._hdr['stamps']
# stc = np.zeros((len(stc_entries)), dtype=object)
# for i in range(1, len(stc_entries)):
#     stc[i-1] = stc_entries[i][1] - stc_entries[i-1][2]
# plt.plot(stc)
#
# # check frame rate
# if (snc_ind < len(snc) - 1) and \
#         (np.abs(snc['frame rate'][snc_ind] / np.nanmean(snc['frame rate']) - 1) > 0.05):
#     if np.isnan(vid['notes'][vid_ind]):
#         vid.loc[vid_ind, 'notes'] = 'sync gap'
#     else:
#         vid.loc[vid_ind, 'notes'] += ', sync gap'
