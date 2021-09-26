
import logging
import asyncio
import aiohttp
import motor.motor_asyncio

from os import environ
from asyncio_throttle import Throttler

from salvo_matteini_etl.loader import upsert_single
from salvo_matteini_etl.query import SearchTweets

logger = logging.getLogger(__name__)


RATE_LIMIT = 1
PERIOD = 30  # in seconds  # 180 / 15 min = 1 / 5 s
MAX_TWEETS = 100
MONGO_DB_NAME = 'twitter'
MONGO_COLL_NAME = 'salvo-matteini-bot'


# initialize env
TWITTER_ACCESS_TOKEN = environ["TWITTER_ACCESS_TOKEN"]
TWITTER_ACCESS_TOKEN_SECRET = environ["TWITTER_ACCESS_TOKEN_SECRET"]
TWITTER_CONSUMER_KEY = environ["TWITTER_CONSUMER_KEY"]
TWITTER_CONSUMER_KEY_SECRET = environ["TWITTER_CONSUMER_KEY_SECRET"]
MONGO_URL = environ["MONGO_URL"]

# throttler
throttler = Throttler(rate_limit=RATE_LIMIT, period=PERIOD)


# @backoff.on_exception(backoff.expo, aiohttp.ClientError,
#                       max_tries=5, max_time=60,
#                       on_backoff=on_backoff_handler)
async def extract(aiohttp_session, since_id, max_tweets):

    query_params = {
        "q": "salvini -filter:media -filter:retweets -filter:native_video -filter:periscope "
             "-filter:vine -filter:images -filter:twimg",
        "result_type": "recent",
        "lang": "it",
        "since_id": since_id,
        "tweet_mode": "extended",
        "count": max_tweets
    }
    q = SearchTweets(query_params)

    async with throttler:
        async with aiohttp_session.get(q.url, headers=q.headers, params=q.params, raise_for_status=True) as resp:
             j = await resp.json()

    return j


# async def transform(j):
#     return j


async def load(motor_client, tweet):
    coll = motor_client[MONGO_DB_NAME][MONGO_COLL_NAME]
    upsert_res = await upsert_single(tweet, coll)
    return upsert_res


async def ETL(aiohttp_client: aiohttp.ClientSession,
              motor_client: motor.motor_asyncio.AsyncIOMotorClient,
              since_id: int):
    """

    :param aiohttp_client:
    :param motor_client:
    :param since_id:
    :return:
    """
    x_res = await extract(aiohttp_client, since_id=since_id, max_tweets=MAX_TWEETS)

    for s in x_res["statuses"]:
        # logger.debug(f"{s['id_str']} - @{s['user']['screen_name']:15s}: {s['full_text']}")
        # t_res = await transform(s)
        await load(motor_client=motor_client, tweet=s)

    return len(x_res["statuses"]), x_res["search_metadata"]["max_id"]


async def batch():
    last_tweet_id = 0
    motor_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL, 27017)
    aiohttp_client = aiohttp.ClientSession()
    while True:
        _, last_tweet_id = await ETL(aiohttp_client, motor_client, since_id=last_tweet_id)
    # await motor_client.close()
    # await aiohttp_client.close()

    # while True:
    #     async with aiohttp.ClientSession() as aiohttp_client:
    #         processed, last_tweet_id = await ETL(aiohttp_client, motor_client, since_id=last_tweet_id)



