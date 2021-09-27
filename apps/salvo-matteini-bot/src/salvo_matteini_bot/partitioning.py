
import logging

from random import shuffle

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
    shuffled_tweets = shuffle(tweets)  # todo add random seed
    # 25 tweets for scoring
    score_tweets = []
    for i in range(25):
        score_tweets.append(shuffled_tweets.pop())
    # train, validation, test sets
    train_n, test_n, valid_n = tuple(int(x * len(shuffled_tweets)) for x in (train, valid, test))

    return shuffled_tweets[:train_n], \
           shuffled_tweets[train_n:train_n+valid_n], \
           shuffled_tweets[train_n+valid_n:], \
           score_tweets