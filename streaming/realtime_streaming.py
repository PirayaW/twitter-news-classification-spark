import sys
import os
import csv
import tweepy
import datetime
import dateutil.parser as parser
import time

python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser


def inRange(time, now, delta):
    begin = now - delta
    return begin < time <= now


class realtimeStreaming:
    def __init__(self, account_list=[], maxnumtweets=100, frequency=30, verbose=False):
        self.account_list = account_list
        self.maxnumtweets = maxnumtweets
        self.verbose = verbose
        self.timedelta = datetime.timedelta(seconds=frequency)

        config = configparser.RawConfigParser()
        config.read('../config.properties')
        self.dest = config.get('Path', 'streaming_unlabelled_data')
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)

        consumer_token = config.get('Twitter', 'consumer_token')
        consumer_secret = config.get('Twitter', 'consumer_secret')
        access_token = config.get('Twitter', 'access_token')
        access_secret = config.get('Twitter', 'access_secret')

        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        self.api = tweepy.API(auth)

    def fetch_realtime(self):
        now = datetime.datetime.utcnow().replace(tzinfo=None)
        if self.verbose:
            print('================== ' + str(now) + ' ==================')
        f, writer = None, None
        for acc in self.account_list:
            for status in tweepy.Cursor(self.api.user_timeline, id=acc).items(self.maxnumtweets):
                tweet = status._json
                tweetTime = parser.parse(tweet['created_at']).replace(tzinfo=None)
                # print(now - self.timedelta)
                # print(tweetTime)
                # print(tweet['text'])
                if (now - self.timedelta) < tweetTime <= now:
                    if f is None:
                        f = open(self.dest + now.strftime("%Y%m%d_%H-%M-%S-%f") + '.csv', 'w')
                        writer = csv.writer(f, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL)
                    text = tweet['text'].encode('ascii', 'replace')
                    row = [text, acc]
                    print(row)
                    writer.writerow(row)
        if f is not None:
            f.close()


def runRealtime():
    maxnumtweets = 100
    frequency_minute = 0.5    # default 15 minutes
    frequency = frequency_minute*60
    timedelta = datetime.timedelta(seconds=frequency)

    config = configparser.RawConfigParser()
    config.read('../config.properties')
    dest = config.get('Path', 'streaming_unlabelled_data')
    if not os.path.exists(dest):
        os.makedirs(dest)

    consumer_token = config.get('Twitter', 'consumer_token')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_token = config.get('Twitter', 'access_token')
    access_secret = config.get('Twitter', 'access_secret')

    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)
    account_list = ['SkyNewsBreak']

    while True:
        now = datetime.datetime.utcnow().replace(tzinfo=None)
        print('================== ' + str(now) + ' ==================')
        for acc in account_list:
            f, writer = None, None
            for status in tweepy.Cursor(api.user_timeline, id=acc).items(maxnumtweets):
                tweet = status._json
                tweetTime = parser.parse(tweet['created_at']).replace(tzinfo=None)
                print(now-timedelta)
                print(tweetTime)
                print(tweet['text'])
                if (now - timedelta) < tweetTime <= now:
                    if writer is None:
                        f = open(dest + now.strftime("%Y%m%d_%H-%M-%S-%f") + '_' + acc + '.csv', 'w')
                        writer = csv.writer(f, delimiter='`', quotechar='|', quoting=csv.QUOTE_ALL)
                    text = tweet['text'].encode('ascii', 'replace')
                    row = [text, acc]
                    print(row)
                    writer.writerow(row)
            if f is not None:
                f.close()
        time.sleep(frequency)


if __name__ == '__main__':
    runRealtime()

