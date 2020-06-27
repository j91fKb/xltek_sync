# Sync Xltek Clinical Recordings with Video

## Dependencies
* >= python3.6

### Install
```
cd <path to this folder>
pip install -r requirements.txt
pip install .
```

<br />

## How to Run
#### Single Xltek Folder
```python
from xltek_sync import Sync
k = Sync(<path to your xltek folder>, output_dir=<optional output directory>)
k.run()
```
#### Entire Subject's Xltek Folder
Assumes that individual xltek folders are direct subfolders of the input folder.
```python
from xltek_sync import run_subject
subject = run_subject(<path to subject xltek folder>, output_dir=<optional output directory>)
```

<br />

## Output:
Output files are stored in the xltek folder in a subfolder named *sync* otherwise to the *output_dir* if passed into *Sync*.
* '..._video_sync.csv' contains the information on the synced videos
* '..._sync_triggers.csv' contains the sync trigger information
* '..._video_frame_sample.csv' contains lookup table for the sample number to each frame of each video
* '..._header.npy' contains the header information

<br />

## Process Description:
This package uses the wonambi python package to read the header information from an xltek folder. The header contains
data on the sync triggers in the format (sample, time which sample occurred) and videos in the format (video name,
start time, end time). It calculates the time difference between the video times and the nearest preceding sync
time and finds the videos sample numbers by using this delta and the calculated relative sample frequency.

#### Example:
*(Dummy numbers are not relative to actual data)*

v1: video start time

v2: video end time

s1: sync start time and sample

s2: next sync start time and sample

sample rate = 1 sample/second throughout

```
video timeline                           v1----------------------------v2
snc timeline     ----s1-------------------------------------s2----------------
relative              |------------------|                   |---------|
                               r1                                r2
```

v1 = 11:00:00am         

v2 = 11:02:00am

s1 = 10:58:30am, sample 100    

s2 = 11:01:30am, sample 280

*so...*

r1 = (v1 - s1) x sample rate = 90 seconds x 1 sample/second = 90 samples

r2 = (v2 - s2) x sample rate = 30 seconds x 1 sample/second = 30 samples

v1 = s1 + r1 = sample 100 + 90 samples = sample 190

v2 = s2 + r2 = sample 280 + 30 samples = sample 310

*thus...*

video starts at sample 190 and ends at sample 310

<br />

## Notes:
- Calculated sampling frequency seems to shift between sync windows by +- .5 Hz.
- There are rare gaps in the *.erd* data where all the values are nan. The initial thousandish values seem to always
be nan.
