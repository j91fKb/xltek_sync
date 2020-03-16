# Sync Xltek Clinical Recordings with Video

## Dependencies
* python > 3.6

### Install with pip
* opencv-python
* matplotlib
* numpy
* pandas
* wonambi
* natsort

Recommended to use virtualenv.

## Run
### Python Console
First run the file..
```python
runfile(<path to this script>)
```
Then run either.
#### Single Xltek Folder
```python
# outputs results into this folder
k = Sync(<path to your xltek folder>, output_dir=<optional output directory>) 
```
#### Entire Subject's Xltek Folder
Assumes that individual xltek folders are one level down from input folder.
```python
# outputs results into this folder
k = run_all(<path to subject xltek folder>, output_dir=<optional output directory>)
```

### Command Line
Input single xltek folder or subject's xltek folder and then follow prompt.
```
# outputs results into this folder
python3 <path to this script> <path to xltek folder> <optional output directory>
```
## Output:
Output files are stored in the xltek folder in a folder called 'sync' unless otherwise specified.
* '..._video_sync.csv' contains the information on the synced videos
* '..._sync_triggers.csv' contains the sync trigger information
* '..._video_frame_sample.csv' contains lookup table for each video frame to the corresponding sample
* '..._header.npy' contains the header information

## Process Description:
This script uses the wonambi python package to read the header information from an xltek folder. The header contains
data on the sync triggers in the format (sample, time which sample occurred) and videos in the format (video name,
start time, end time). It then calculates the time difference between the video times and the nearest preceding sync
time and finds the videos sample numbers by using this distance and the relative sample frequency.

### Example:
(Dummy numbers are not relative to actual data)

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

so...

r1 = (v1 - s1) x sample rate = 90 seconds x 1 sample/second = 90 samples

r2 = (v2 - s2) x sample rate = 30 seconds x 1 sample/second = 30 samples

v1 = s1 + r1 = sample 100 + 90 samples = sample 190

v2 = s2 + r2 = sample 280 + 30 samples = sample 310

thus...

video starts at sample 190 and ends at sample 310

## Notes:
- You can imagine there are various alignments between the videos and the sync triggers so the sync method tries to
cover all of these.
- Calculated sampling frequency seems to shift between sync windows by +- .5 Hz.
- There are rare gaps in the erd data where all the values are nan. The initial thousandish values seem to always
be nan.
