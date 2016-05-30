import sys
import os
import shutil
import time
import glob
from realtime_streaming import realtimeStreaming

python_version = sys.version_info.major
if python_version == 3:
    import configparser
    from Tools.scripts.treesync import raw_input
else:
    import ConfigParser as configparser

if __name__ == '__main__':
    frequency = 30  # default frequency
    if len(sys.argv) > 1:
        frequency = int(sys.argv[1])

    config = configparser.RawConfigParser()
    config.read('../config.properties')

    path = config.get('Path', 'batch_data')
    dest = config.get('Path', 'streaming_data')
    test_dest = config.get('Path', 'streaming_test_data')
    signal_dest = config.get('Path', 'streaming_signal')
    path_unlabelled = config.get('Path', 'batch_unlabelled')
    unlabelled_dest = config.get('Path', 'streaming_unlabelled_data')

    # list of files in source folders
    filenames = next(os.walk(path))[2]
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    filenames.sort()
    del filenames[15:29]
    print(filenames)

    unlabelled_filenames = next(os.walk(path_unlabelled))[2]
    if '.DS_Store' in unlabelled_filenames:
        unlabelled_filenames.remove('.DS_Store')
    unlabelled_filenames.sort()

    # create streaming folder if it's not existed
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)
    if not os.path.exists(unlabelled_dest):
        os.makedirs(unlabelled_dest)
    if not os.path.exists(signal_dest):
        os.makedirs(signal_dest)

    # remove any files existed in the destination folders
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    shutil.rmtree(test_dest, ignore_errors=True)
    os.makedirs(test_dest)
    shutil.rmtree(signal_dest, ignore_errors=True)
    os.makedirs(signal_dest)
    # remove only batch files in streaming unlabelled folder
    # batch data's filenames are in this format: dddddddd_dd.csv where d is a digit
    for files in glob.glob(unlabelled_dest + ('[0-9]' * 8) + '_' + ('[0-9]' * 2) + '.csv'):
        os.remove(files)
    print('Deleted existing files')

    ## streaming options
    # stream batch train/test data or not
    if python_version == 3:
        inputtest = input("Do you want to update the model? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to update the model? (y/n) : \n")
    train = inputtest == 'Y' or inputtest == 'y'

    # stream batch unlabelled data or not
    if python_version == 3:
        inputtest = input("Do you want to stream batch unlabelled data? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to stream batch unlabelled data? (y/n) : \n")
    unlabel = inputtest == 'Y' or inputtest == 'y'

    # stream realtime data or not
    if python_version == 3:
        inputtest = input("Do you want to stream realtime tweets? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to stream realtime tweets? (y/n) : \n")
    realtime = inputtest == 'Y' or inputtest == 'y'

    # make sure everything is ready before streaming
    time.sleep(frequency)

    realtimeStreaming = realtimeStreaming(['SkyNewsBreak'], 10, frequency, verbose=True)

    test_file = filenames[len(filenames) - 1]
    count, i, j = 0, 0, 0
    while True:
        if realtime:
            realtimeStreaming.fetch_realtime()
        if unlabel and j < len(unlabelled_filenames):
            shutil.copyfile(path_unlabelled + unlabelled_filenames[j], unlabelled_dest + unlabelled_filenames[j])
            print('Copied Unlabeled file: ' + unlabelled_filenames[j])
            j += 1
        if train and i < len(filenames):
            if (count > 0 and count % 3 == 0) or i == len(filenames) - 1:
                shutil.copyfile(path + test_file, test_dest + str(count) + '_' + test_file)
                print('Copied Testing file: ' + test_file)
                if i == len(filenames) - 1:
                    i += 1
            else:
                shutil.copyfile(path + filenames[i], dest + filenames[i])
                print('Copied Training file: ' + filenames[i])
                i += 1
            count += 1
        if train:
            if not realtime and i >= len(filenames) and not (unlabel and j < len(unlabelled_filenames)):
                time.sleep(frequency)
                print('sending save signal')
                with open(signal_dest + 'signal.txt', 'w') as f:
                    f.write('save')
                exit(0)
        else:
            if not realtime and not (unlabel and j < len(unlabelled_filenames)):
                exit(0)
        time.sleep(frequency)

