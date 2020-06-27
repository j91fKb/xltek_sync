import os
from natsort import natsorted
from pathlib import Path
from .sync import Sync


# Run on a subjects entire folder
def run_subject(pt_xltek_dir, output_dir=None):
    pt_xltek_dir = Path(pt_xltek_dir)

    # grab xltek subfolders
    sub_dirs = natsorted([(pt_xltek_dir / f)
                          for f in os.listdir(pt_xltek_dir) if os.path.isdir(pt_xltek_dir / f)])

    # main loop through all subfolders
    any_failed = False
    all_data = {}
    for sub_dir in sub_dirs:
        print('Working on', sub_dir)
        sub_dir_name = os.path.split(sub_dir)[-1]

        # make a sub directory in the output folder for each xltek folder
        if output_dir:
            output_sub_dir = Path(output_dir) / sub_dir_name
            if not os.path.isdir(output_sub_dir):
                os.mkdir(output_sub_dir)
        else:
            output_sub_dir = None

        # sync up
        try:
            all_data[sub_dir_name] = Sync(sub_dir, output_sub_dir)
            all_data[sub_dir_name].run()
        except:
            all_data[sub_dir_name] = 'Failed'
            any_failed = True

        print("==============================================")

    # print any failed folders
    if any_failed:
        print('The following folders failed:')
        for name, output in all_data.items():
            if output == 'Failed':
                print(name)

    return all_data
