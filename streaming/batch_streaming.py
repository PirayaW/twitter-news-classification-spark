import sys
import os
import shutil
import time

# from Tools.scripts.treesync import raw_input
from random import sample

python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser

if __name__ == '__main__':
    frequency = 20  # default frequency
    if len(sys.argv) > 1:
        frequency = int(sys.argv[1])

    config = configparser.RawConfigParser()
    config.read('../config.properties')

    path = config.get('Path', 'batch_data')
    dest = config.get('Path', 'streaming_data')
    test_dest = config.get('Path', 'streaming_test_data')

    # list of files
    filenames = next(os.walk(path))[2]
    filenames.sort()
    if not os.path.exists(dest):
        os.makedirs(dest)
    destfiles = next(os.walk(dest))[2]
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)
    desttestfiles = next(os.walk(test_dest))[2]

    if len(destfiles) > 0:
        if python_version == 3:
            delete = input("Destination folder is not empty. Delete? (y/n): \n")
        else:
            delete = raw_input("Destination folder is not empty. Delete? (y/n): \n")
        if delete == 'y' or delete == 'Y':
            shutil.rmtree(dest, ignore_errors=True)
            os.makedirs(dest)
            shutil.rmtree(test_dest, ignore_errors=True)
            os.makedirs(test_dest)
        else:
            print("Terminating")
            exit(0)

    tr_percent = 0.7
    tr_numfiles = int(round(tr_percent*len(filenames)))
    tr_num = sample(range(len(filenames)), tr_numfiles)

    for i in range(0, len(filenames)):
        file = filenames[i]
        if (i in tr_num or i == 0) and i < len(filenames)-1:
            shutil.copyfile(path + file, dest + file)
            print('Copied Training file: ' + file)
            time.sleep(frequency)
        else:
            shutil.copyfile(path + file, test_dest + file)
            print('Copied Testing file: ' + file)
            time.sleep(frequency)