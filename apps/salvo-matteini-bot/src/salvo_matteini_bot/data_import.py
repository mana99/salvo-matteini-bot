
import logging
import pandas as pd

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
