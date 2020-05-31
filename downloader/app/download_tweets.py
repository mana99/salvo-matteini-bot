
import concurrent.futures
import pymongo
from pymongo.write_concern import WriteConcern
import tweepy
import configparser
import logging
import logging.config
#import numpy as np
from os.path import dirname, abspath, join

CURRENT_DIR = dirname(abspath(__file__))
INI_PATH = join(CURRENT_DIR, '.ini')
LOGGER_PATH = join(CURRENT_DIR, 'logger.conf')
print(INI_PATH)
print(LOGGER_PATH)

conf = configparser.ConfigParser()
conf.read(INI_PATH)

logging.config.fileConfig(fname=LOGGER_PATH)
logger = logging.getLogger("downloader")

# twitter authentication
auth = tweepy.OAuthHandler(conf['twitter']['consumer_key'], conf['twitter']['consumer_key_secret'])
auth.set_access_token(conf['twitter']['access_token'], conf['twitter']['access_token_secret'])
api = tweepy.API(auth, wait_on_rate_limit=True)

# mongo authentication
#conn_str = f"mongodb://{conf['mongo']['user']}:{conf['mongo']['pass']}@{conf['mongo']['host']}"
conn_str = "mongodb://{}:{}@{}".format(conf['mongo']['user'],
                                       conf['mongo']['pass'],
                                       conf['mongo']['host'])
db = pymongo.MongoClient(conn_str)[conf['mongo']['db']]
coll = db.get_collection(conf['mongo']['collection'])


class mresults:
    SKIPPED  = 1, 0, 0, 0
    UPDATED  = 0, 1, 0, 0
    INSERTED = 0, 0, 1, 0
    FAILED   = 0, 0, 0, 1


def upsert_single(tweet_dict):

    coll_w = coll.with_options(write_concern=WriteConcern(w='majority'))
    try:
        upsert_result = coll_w.update_one(filter={'id': tweet_dict['id']},
                                          update={'$set': tweet_dict},
                                          upsert=True)
        r = upsert_result.raw_result
        if r['n'] == 1 and r['ok'] == 1:
            if r['updatedExisting']:
                if r['nModified'] == 1:
                    logger.debug(f"{tweet_dict['id_str']} - Updated")
                    return mresults.UPDATED
                else:
                    logger.debug(f"{tweet_dict['id_str']} - Skipped")
                    return mresults.SKIPPED
            elif r['upserted']:
                logger.debug(f"{tweet_dict['id_str']} - Inserted")
                return mresults.INSERTED
        logger.error(f"{tweet_dict['id_str']} - Failed")
        return mresults.FAILED
    except OverflowError as e:
        logger.error(f"{tweet_dict['id_str']} - OverflowError ({e})")
        return mresults.FAILED
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.error(f"{tweet_dict['id_str']} - ServerSelectionTimeoutError ({e})")
        return mresults.FAILED


words = 'salvini -filter:retweets'
date_from = "2020-01-01"
max_tweets = 100

tweets = tweepy.Cursor(api.search,
                       q=words, lang="it", since=date_from, tweet_mode='extended').items(max_tweets)

MAX_WORKERS = 5

try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
       map_out = executor.map(upsert_single, (t._json for t in tweets))
except tweepy.error.RateLimitError as e:
    logger.error(e)


from functools import reduce

skipped, updated, inserted, failed = reduce(lambda x, y: (x[0]+y[0],
                                                          x[1]+y[1],
                                                          x[2]+y[2],
                                                          x[3]+y[3]), list(map_out))
# summary = np.array(list(map_out)).sum(axis=0)
# skipped, updated, inserted, failed = tuple(summary)
logger.info(f"skipped: {skipped}, updated: {updated}, inserted: {inserted}, failed: {failed}")
