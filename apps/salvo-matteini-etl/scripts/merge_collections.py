
import logging
import pymongo
import motor.motor_asyncio
import asyncio

MONGO_URL = "mongodb://127.0.0.1:27017"

logging.root.setLevel(logging.DEBUG)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
# client = pymongo.MongoClient(MONGO_URL)

target_coll = client["twitter"]["salvo-matteini-bot"]
source_colls = [
    client["tweets"]["tweets"],
    client["tweets"]["dump_tweets_2020-06-08"]
]

# for source_coll in source_colls:
#     for tweet in source_coll.find({}):
#         find_res = target_coll.find_one(filter={'id': tweet['id']})
#         if not find_res:
#             ins_res = target_coll.insert_one(tweet)
#             if ins_res.acknowledged:
#                 logging.info(f"{tweet['id_str']} - INSERTED ({ins_res.inserted_id})")
#                 del_res = source_coll.delete_one(filter={'_id': tweet['_id']})
#                 if del_res.acknowledged:
#                     logging.info(f"{tweet['id_str']} - DELETED")
#             else:
#                 logging.error(f"{tweet['id_str']} - NOT INSERTED")
#         else:
#             logging.info(f"{tweet['id_str']} - SKIPPED")
#             del_res = source_coll.delete_one(filter={'_id': tweet['_id']})
#             if del_res.acknowledged:
#                 logging.info(f"{tweet['id_str']} - DELETED")


async def find_insert_delete(tweet, source_coll, target_coll):
    logging.info(tweet["id"])
    find_res = await target_coll.find_one(filter={'id': tweet['id']})
    if not find_res:
        ins_res = await target_coll.insert_one(tweet)
        if ins_res.acknowledged:
            logging.info(f"{tweet['id_str']} - INSERTED ({ins_res.inserted_id})")
            del_res = await source_coll.delete_one(filter={'_id': tweet['_id']})
            if del_res.acknowledged:
                logging.info(f"{tweet['id_str']} - DELETED")
        else:
            logging.error(f"{tweet['id_str']} - NOT INSERTED")
    else:
        logging.info(f"{tweet['id_str']} - SKIPPED")
        del_res = await source_coll.delete_one(filter={'_id': tweet['_id']})
        if del_res.acknowledged:
            logging.info(f"{tweet['id_str']} - DELETED")


async def main():
    for source_coll in source_colls:
        while True:
            tasks = [
                asyncio.ensure_future(find_insert_delete(tweet, source_coll, target_coll))
                async for tweet in source_coll.find({}).limit(500)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

loop = asyncio.get_event_loop()
tweets = loop.run_until_complete(main())

