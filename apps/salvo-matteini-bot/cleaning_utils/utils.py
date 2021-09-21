import preprocessor as p
import csv
import pandas as pd

path_to_tweet_csv = "tweets/tweets_csv_test.csv"
EMOJI_PREFIX = '/EMOJI'
HASHTAG_PREFIX = '/HASHTAG'
URL_PREFIX = '/URL'
MENTIONS_PREFIX = '/MENTION'
TWEET_COLUMN_NAME = 'text'


def tweet_parsing(current_tweet):

    # create the parser
    parser_tweet = p.parse(current_tweet)

    # check and substitute for the element we do not want
    if parser_tweet.emojis:
        for emoji in parser_tweet.emojis:
            emoji_str = emoji.match
            current_tweet = current_tweet.replace(emoji_str, EMOJI_PREFIX)

    if parser_tweet.hashtags:
        for hashtag in parser_tweet.hashtags:
            hashtag_str = hashtag.match
            # current_tweet = current_tweet.replace(hashtag_str, '{}{}'.format(HASHTAG_PREFIX, hashtag_str))
            current_tweet = current_tweet.replace(hashtag_str, hashtag_str[1:])

    if parser_tweet.urls:
        for url in parser_tweet.urls:
            url_str = url.match
            # invert comment to include the url
            #current_tweet = current_tweet.replace(url_str, '{}{}'.format(URL_PREFIX, url_str))
            current_tweet = current_tweet.replace(url_str, URL_PREFIX)

    if parser_tweet.mentions:
        for mention in parser_tweet.mentions:
            mention_str = mention.match
            # invert comment to include the url
            #current_tweet = current_tweet.replace(mention_str, '{}{}'.format(MENTIONS_PREFIX, mention_str))
            current_tweet = current_tweet.replace(mention_str, MENTIONS_PREFIX)

    # cleaned tweet
    print(current_tweet)
    # Preprocessor is /HASHTAG#awesome 👍 /URLhttps://github.com/s/preprocessor
    return current_tweet


def file_cleaning(path_to_tweet_csv):
    result_csv = 'new_{}'.format(path_to_tweet_csv)
    pd.read_csv(path_to_tweet_csv, nrows=1).head(0).to_csv(result_csv)

    dataframe = pd.read_csv(path_to_tweet_csv, iterator=True, chunksize=10000)

    for chunk in dataframe:
        chunk[TWEET_COLUMN_NAME] = chunk[TWEET_COLUMN_NAME].apply(tweet_parsing)
        chunk.to_csv(result_csv, mode='a', header=None)


# file_cleaning(path_to_tweet_csv)


if __name__ == '__main__':
    tweet_parsing('Preprocessor is #awesome 👍 https://github.com/s/preprocessor')