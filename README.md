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

## Run
### Python Console
```python
s = "Python syntax highlighting"
print s
```
```
runfile(<path to this script>)
```
#### Single Xltek Folder
```
k = Sync(<path to your xltek folder>) #outputs results into this folder

k = Sync(<path to your xltek folder>, <path to output folder>) # optional output directory
```
#### Entire Subject's Xltek Folder
Assumes that xltek folders are within this folder

### Command Line
python3 <path to this script> <path to subject's xltek folder>

