
import logging
from typing import List

import pandas as pd
import pymongo

from salvo_matteini_bot import MONGO_URL, MONGO_DB_NAME, MONGO_COLL_NAME

logger = logging.getLogger(__name__)


def import_data(fn: str, n_max: int = None) -> pd.DataFrame:
    """
    Import tweets from CSV file.

    :param fn: CSV file path
    :param n_max: max number of rows to import (import full file if not specified)
    :return: full_text column of tweets
    """
    logger.info("Importing")
    if n_max:
        return pd.read_csv(fn, usecols=['full_text'], nrows=n_max)
    else:
        return pd.read_csv(fn, usecols=['full_text'])
    # for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
    #      print(chunk)


def mongo_import(limit: int = 1000) -> List[str]:
    client = pymongo.MongoClient(MONGO_URL)
    coll = client[MONGO_DB_NAME][MONGO_COLL_NAME]
    cursor = coll.find().limit(limit)
    #for doc in cursor:
    #    yield doc["full_text"]
    tweets = [doc["full_text"] for doc in cursor]
    return tweets
