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
    datapath = config.get('Path', 'json_data')
    batchpath = config.get('Path', 'batch_data')

    account = ['CNNPolitics',
               'YahooFinance',
               'YahooSports',
               # 'weatherchannel',
               'TechCrunch',
               # 'ForbesTech',
               'ScienceNews',
               'HuffPostCrime',
               'CrimeInTheD',
               'CNNent',
               'YahooCelebrity',
               'YahooMusic']
    label = [0,
             1,
             2,
             # 'weather',
             3,
             # 3,
             3,
             5,
             5,
             4,
             4,
             4]
    begin = toDatetime('Sun May 01 00:00:00 +0000 2016')
    end = toDatetime('Sat May 07 23:59:59 +0000 2016')
    delta = datetime.timedelta(hours=12)
    times = []
    date = begin
    while date <= end:
        times.append(date)
        date = date + delta
    beforeTime = begin - delta

    files = []
    writers = []
    for date in times:
        file = open(batchpath + date.strftime("%Y%m%d_%H") + '.csv', 'w')
        files.append(file)
        writers.append(csv.writer(file, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL))

    for i in range(len(account)):
        acc = account[i]
        count = 0
        f = open(datapath + acc + '_oneweek.json')
        for line in f:
            tweet = json.loads(line)
            time = toDatetime(tweet['created_at'])
            # can be optimized
            current = begin
            index = 0
            while time >= current:
                current = current + delta
                index += 1
            row = [tweet['text'].encode('ascii', 'replace'), label[i], time.strftime("%Y%m%d_%H"), acc]
            writers[index - 1].writerow(row)
        f.close()

    for file in files:
        file.close()

    # tweets before begin time
    beforeFile = open(batchpath + beforeTime.strftime("%Y%m%d_%H") + '.csv', 'w')
    beforeWriter = csv.writer(beforeFile, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL)
    for i in range(len(account)):
        acc = account[i]
        f = open(datapath + acc + '_before.json')
        for line in f:
            tweet = json.loads(line)
            time = toDatetime(tweet['created_at'])
            row = [tweet['text'].encode('ascii', 'replace'), label[i], time.strftime("%Y%m%d_%H"), acc]
            beforeWriter.writerow(row)
        f.close()

    beforeFile.close()

# file = open(batchpath + times[len(files)-1].strftime("%Y%m%d_%H") + '.csv')
# reader = csv.reader(file, delimiter='`', quotechar='|')
# for row in reader:
# 	print(row[0])   # text
# 	print(row[1])   # label
