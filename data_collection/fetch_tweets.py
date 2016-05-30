### HOW TO RUN:
# python fetch_tweets.py <username> <max_number_of_tweets>

import sys
import tweepy
from tweepy import OAuthHandler
import json

python_version = sys.version_info.major
if python_version == 3:
    import configparser
else:
    import ConfigParser as configparser


def twitter_fetch(screen_name="", maxnumtweets=0):
    maxnumtweets = int(maxnumtweets)

    config = configparser.RawConfigParser()
    config.read('../config.properties')

    datapath = config.get('Path', 'json_data')

    consumer_token = config.get('Twitter', 'consumer_token')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_token = config.get('Twitter', 'access_token')
    access_secret = config.get('Twitter', 'access_secret')

    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth)

    filename = datapath + 'new_' + screen_name + '.json'
    with open(filename, 'a') as outfile:
        for status in tweepy.Cursor(api.user_timeline, id=screen_name).items(maxnumtweets):
            json_str = status._json
            json.dump(json_str, outfile)
            outfile.write("\n")
    outfile.close()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        screen_name = sys.argv[1]
        maxnumtweets = sys.argv[2]
        twitter_fetch(screen_name, maxnumtweets)
    else:
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
        for acc in account:
            twitter_fetch(acc, 1000)
