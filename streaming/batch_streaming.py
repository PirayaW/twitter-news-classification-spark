import sys
import os
import shutil
import time

from Tools.scripts.treesync import raw_input
from random import sample, randint

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
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    filenames.sort()
    if not os.path.exists(dest):
        os.makedirs(dest)
    destfiles = next(os.walk(dest))[2]
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)
    desttestfiles = next(os.walk(test_dest))[2]

    if python_version == 3:
        inputtest = input("Do you want to run test for only one file? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to run test for only one file? (y/n) : \n")

    if len(destfiles) > 0:
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        shutil.rmtree(test_dest, ignore_errors=True)
        os.makedirs(test_dest)
        print('Deleted existing files')
        time.sleep(frequency)

    count = 0
    test_file_found = False
    tr_percent = 0.7
    tr_numfiles = int(round(tr_percent*len(filenames)))
    tr_num = sample(range(len(filenames)), tr_numfiles)
    test_num = set(range(len(filenames))) - set(tr_num)
    test_num = list(test_num)
    print(tr_num)
    print(test_num)

    if 0 in test_num:
        test_num.remove(0)
        print(test_num)

    # COPY THE FIRST FILE INDEX 0
    shutil.copyfile(path + filenames[0], dest + filenames[0])
    print('Copied Training file: ' + filenames[0])
    time.sleep(frequency)
    count += 1
    ###

    for i in tr_num:
        file = filenames[i]
        if i != 0 and i < len(filenames)-1:
            shutil.copyfile(path + file, dest + file)
            print('Copied Training file: ' + file)
            time.sleep(frequency)

            count += 1
            if count % 3 == 0 and (inputtest == 'Y' or inputtest =='y'):
                if test_file_found == False:
                    j = randint(0,len(test_num)-1)
                    test_file = filenames[test_num[j]]
                    test_file_found = True
                shutil.copyfile(path + test_file, test_dest + test_file)
                print('Copied Testing file: ' + test_file)
                time.sleep(frequency)

    if inputtest != 'Y' and inputtest != 'y':
        # COPY THE LAST FILE AT INDEX LAST
        file = filenames[len(filenames)-1]
        shutil.copyfile(path + file, test_dest + file)
        print('Copied Testing file: ' + file)
        time.sleep(frequency)
        ###
        for i in test_num:
            if i != len(filenames)-1:
                file = filenames[i]
                shutil.copyfile(path + file, test_dest + file)
                print('Copied Testing file: ' + file)
                time.sleep(frequency)
