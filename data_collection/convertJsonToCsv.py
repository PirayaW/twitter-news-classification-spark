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

    # account = ['new_bpolitics',
    #            'new_ReutersBiz',
    #            'new_espn'
    #            ]
    # label = [0,
    #          1,
    #          2
    #          ]
    account = ['new_CrimeStoppersOR',
               'new_CrimeWorId',
               'new_e_entertainment',
               'new_TMZ',
               'new_ftfinancenews',
               'new_nytpolitics',
               'new_sciam'
               ]
    label = [5,
             5,
             4,
             4,
             1,
             0,
             3
             ]

    file = open('additionalEntertainmentPoliticsFinanceSciCrime.csv', 'w')
    writer = csv.writer(file, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL)

    for i in range(len(account)):
        acc = account[i]
        f = open(datapath + acc + '.json')
        for line in f:
            tweet = json.loads(line)
            time = toDatetime(tweet['created_at'])
            row = [tweet['text'].encode('ascii', 'replace'), label[i], time.strftime("%Y%m%d_%H"), acc]
            writer.writerow(row)
    f.close()

