import sys
import os
import shutil
import time
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

    # list of files
    filenames = next(os.walk(path))[2]
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    filenames.sort()
    del filenames[15:29]
    print(filenames)
    if not os.path.exists(dest):
        os.makedirs(dest)
    destfiles = next(os.walk(dest))[2]
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)
    desttestfiles = next(os.walk(test_dest))[2]
    if not os.path.exists(signal_dest):
        os.makedirs(signal_dest)

    if python_version == 3:
        inputtest = input("Do you want to update the model? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to update the model? (y/n) : \n")

    train = inputtest == 'Y' or inputtest == 'y'

    if python_version == 3:
        inputtest = input("Do you want to stream realtime tweets? (y/n) : \n")
    else:
        inputtest = raw_input("Do you want to stream realtime tweets? (y/n) : \n")

    realtime = inputtest == 'Y' or inputtest == 'y'

    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    shutil.rmtree(test_dest, ignore_errors=True)
    os.makedirs(test_dest)
    shutil.rmtree(signal_dest, ignore_errors=True)
    os.makedirs(signal_dest)
    print('Deleted existing files')

    time.sleep(frequency)

    realtimeStreaming = realtimeStreaming(['SkyNewsBreak'], 10, frequency, verbose=True)

    test_file = filenames[len(filenames) - 1]
    count, i = 0, 0
    while True:
        if realtime:
            realtimeStreaming.fetch_realtime()
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
        if not realtime and not (train and i < len(filenames)):
            time.sleep(frequency)
            print('sending save signal')
            with open(signal_dest + 'signal.txt', 'w') as f:
                f.write('save')
            exit(0)
        time.sleep(frequency)
