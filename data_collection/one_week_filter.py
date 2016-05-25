import dateutil.parser as parser
import json
import sys


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

    account = ['CNNPolitics',
               'YahooFinance',
               'YahooSports',
               # 'weatherchannel',
               'TechCrunch',
               'ForbesTech',
               'ScienceNews',
               'HuffPostCrime',
               'CrimeInTheD',
               'CNNent',
               'YahooCelebrity'
               # 'YahooMusic'
               ]
    begin = toDatetime('Sun May 15 00:00:00 +0000 2016')
    end = toDatetime('Sat May 21 23:59:59 +0000 2016')
    interval = end - begin
    for acc in account:
        count = 0
        f = open(datapath + 'new_' + acc + '.json')
        out = open(datapath + acc + '_oneweek2.json', 'w')
        # before = open(datapath + acc + '_before.json', 'w')
        for line in f:
            tweet = json.loads(line)
            time = toDatetime(tweet['created_at'])
            if begin <= time <= end and tweet['in_reply_to_status_id'] is None:
                json.dump(tweet, out)
                out.write("\n")
                count += 1
            # elif time < begin and tweet['in_reply_to_status_id'] is None:
            #     json.dump(tweet, before)
            #     before.write("\n")
        out.close()
        f.close()
        # before.close()
        print('============ ' + acc + ' ================')
        print("#Tweets last week: %d" % count)
        print("Rate per day: %.2f" % (count / 7.0))
