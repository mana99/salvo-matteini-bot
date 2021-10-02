
import logging
import random

from random import shuffle

from salvo_matteini_bot import RANDOM_STATE

logger = logging.getLogger(__name__)


def partition_tweets(tweets, train_valid_test):
    """
    Perform holdout partitioning by shuffling tweets and splitting according to the train:test ratio.

    :param tweets:
    :param train_valid_test:
    :return:
    """

    train, valid, test = train_valid_test

    if train + valid + test != 1:
        train, valid, test = tuple(x / (train + valid + test) for x in (train, valid, test))

    logger.info("Shuffling tweets...")
    random.Random(RANDOM_STATE).shuffle(tweets)
    # 25 tweets for scoring
    score_tweets = []
    for i in range(25):
        score_tweets.append(tweets.pop())
    # train, validation, test sets
    train_n, test_n, valid_n = tuple(int(x * len(tweets)) for x in (train, valid, test))

    output = {
        "train": tweets[:train_n],
        "validation": tweets[train_n:train_n+valid_n],
        "test": tweets[train_n+valid_n:],
        "score": score_tweets
    }

    return output