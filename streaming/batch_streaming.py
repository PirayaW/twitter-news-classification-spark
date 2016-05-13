import sys
import os
import shutil
import time

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

    # list of files
    filenames = next(os.walk(path))[2]
    filenames.sort()
    if not os.path.exists(dest):
        os.makedirs(dest)
    destfiles = next(os.walk(dest))[2]

    if len(destfiles) > 0:
        if python_version == 3:
            delete = input("Destination folder is not empty. Delete? (y/n): \n")
        else:
            delete = raw_input("Destination folder is not empty. Delete? (y/n): \n")
        if delete == 'y' or delete == 'Y':
            shutil.rmtree(dest, ignore_errors=True)
            os.makedirs(dest)
        else:
            print("Terminating")
            exit(0)

    for file in filenames:
        shutil.copyfile(path + file, dest + file)
        print('Copied: ' + file)
        time.sleep(frequency)