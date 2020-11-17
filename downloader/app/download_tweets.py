
import logging
import pymongo
import tweepy
import time

from dotenv import load_dotenv
from os import environ
from pymongo.write_concern import WriteConcern

MAX_REQUESTS = 180
TIME_WINDOW_MINS = 15
DB_NAME = 'twitter'
COLLECTION_NAME = 'salvo-matteini-bot'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
load_dotenv()


def init() -> (tweepy.API, pymongo.collection.Collection):

    # twitter authentication
    auth = tweepy.OAuthHandler(environ['TWITTER_CONSUMER_KEY'], environ['TWITTER_CONSUMER_KEY_SECRET'])
    auth.set_access_token(environ['TWITTER_ACCESS_TOKEN'], environ['TWITTER_ACCESS_TOKEN_SECRET'])
    api = tweepy.API(auth)  #, wait_on_rate_limit=True)   # wait on rate limit when sending requests

    logger.debug(f"API version: {api.api_root[1:]}")
    logger.debug(f"Cache: {api.cache}")
    logger.debug(f"Timeout: {api.timeout}")

    # mongo authentication
    coll = pymongo.MongoClient(environ['MONGO_CONNECTION_STRING'])[DB_NAME].get_collection(COLLECTION_NAME)

    return api, coll


def upsert_single(tweet: dict, coll: pymongo.collection.Collection):

    class mresults:
        SKIPPED  = 1, 0, 0, 0
        UPDATED  = 0, 1, 0, 0
        INSERTED = 0, 0, 1, 0
        FAILED   = 0, 0, 0, 1

    coll_w = coll.with_options(write_concern=WriteConcern(w='majority'))
    try:
        upsert_result = coll_w.update_one(filter={'id': tweet['id']},
                                          update={'$set': tweet},
                                          upsert=True)
        r = upsert_result.raw_result
        if r['n'] == 1 and r['ok'] == 1:
            if r['updatedExisting']:
                if r['nModified'] == 1:
                    logger.debug(f"{tweet['id_str']} - Updated")
                    return mresults.UPDATED
                else:
                    logger.debug(f"{tweet['id_str']} - Skipped")
                    return mresults.SKIPPED
            elif r['upserted']:
                logger.debug(f"{tweet['id_str']} - Inserted")
                return mresults.INSERTED
        logger.error(f"{tweet['id_str']} - Failed")
        return mresults.FAILED
    except OverflowError as e:
        logger.error(f"{tweet['id_str']} - OverflowError ({e})")
        return mresults.FAILED
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.error(f"{tweet['id_str']} - ServerSelectionTimeoutError ({e})")
        return mresults.FAILED


def scan(api: tweepy.API, since_id: int = 0):

    logger.debug("Twitter API search request")

    return tweepy.Cursor(api.search,
                         # q="salvini -filter:retweets",
                         q="salvini -filter:media -filter:retweets -filter:native_video -filter:periscope "
                           "-filter:vine -filter:images -filter:twimg",
                         result_type='recent',
                         lang="it",
                         since_id=since_id,
                         # count=100,
                         # since=date_from,
                         tweet_mode='extended')


def main():

    api, coll = init()
    started_at = time.time()
    max_id = next(coll.aggregate([
        {'$group': {'_id': None, 'max_id': {'$max': '$id'}}}
    ]))['max_id']

    # max requests event
    def on_max_requests():
        nonlocal started_at
        wait_seconds = TIME_WINDOW_MINS * 60 - (time.time() - started_at)
        logger.warning(f"Too many requests. Waiting {int(wait_seconds/60)} minutes...")
        time.sleep(wait_seconds)
        started_at = time.time()

    # main loop
    while True:
        tweets = scan(api, since_id=max_id)
        try:
            for i, t in enumerate(tweets.items()):
                if t.id > max_id:
                    max_id = t.id
                res = upsert_single(t._json, coll)
                logger.debug(f"{t.id_str} - {res}")
        except tweepy.error.RateLimitError:
            on_max_requests()
        except tweepy.error.TweepError:
            on_max_requests()


if __name__ == '__main__':
    main()


# import concurrent.futures
# from functools import reduce
# with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#     map_out = executor.map(upsert_single, (t._json for t in tweets), (coll for t in tweets))
#     skipped, updated, inserted, failed = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]),
#                                                 list(map_out))
#     logger.info(f"skipped: {skipped}, updated: {updated}, inserted: {inserted}, failed: {failed}")

