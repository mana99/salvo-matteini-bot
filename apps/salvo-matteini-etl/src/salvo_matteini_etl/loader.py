
import logging
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import WriteConcern

logger = logging.getLogger(__name__)


async def upsert_single(tweet: dict, coll: AsyncIOMotorCollection):

    coll_w = coll.with_options(write_concern=WriteConcern(w='majority'))
    upsert_result = await coll_w.update_one(filter={'id': tweet['id']},
                                            update={'$set': tweet},
                                            upsert=True)

    # logging
    r = upsert_result.raw_result
    if r['n'] == 1 and r['ok'] == 1:
        if r['updatedExisting']:
            if r['nModified'] == 1:
                logger.debug(f"{tweet['id_str']} - Updated")
            else:
                logger.debug(f"{tweet['id_str']} - Skipped")
        elif r['upserted']:
            logger.debug(f"{tweet['id_str']} - Inserted")
    else:
        logger.error(f"{tweet['id_str']} - Failed")

    return upsert_result

    # except OverflowError as e:
    #     logger.error(f"{tweet['id_str']} - OverflowError ({e})")
    # except pymongo.errors.ServerSelectionTimeoutError as e:
    #     logger.error(f"{tweet['id_str']} - ServerSelectionTimeoutError ({e})")
