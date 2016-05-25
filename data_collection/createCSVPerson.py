import json
import datetime
import csv
import sys
import dateutil.parser as parser

python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser


def toDatetime(str):
    return parser.parse(str)


if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read('../config.properties')
    datapath = config.get('Path', 'json_persondata')
    batchpath = config.get('Path', 'persondata')

    account = ['jaketapper',
                   'YahooNoise',
                  'DanWetzel',
                   'DaveDiMartino',
                   'michaelsantoli',
                   'verge']

    for i in range(len(account)):
        acc = account[i]
        f = open(datapath + acc + '.json')
        file = open(batchpath + acc + '.csv', 'w',newline='')
        writer = csv.writer(file, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL)
        for line in f:
            tweet = json.loads(line)
            row = [tweet['text'].encode('ascii', 'replace')]
            writer.writerow(row)
        f.close()
        file.close()

# file = open(batchpath + times[len(files)-1].strftime("%Y%m%d_%H") + '.csv')
# reader = csv.reader(file, delimiter='`', quotechar='|')
# for row in reader:
# 	print(row[0])   # text
# 	print(row[1])   # label
