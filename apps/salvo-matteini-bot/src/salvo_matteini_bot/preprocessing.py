
import itertools
import logging
import nltk
import preprocessor  # todo remove (tweet-preprocessor)
import re

from numpy import unicode
from keras.preprocessing.text import Tokenizer

from salvo_matteini_bot import SEQ_LENGTH, START_TOKEN

logger = logging.getLogger(__name__)


def preprocess(tweets):
    """

    1. Filter tweets with too many words
    2. Lowercase words
    3. Remove punctuation
    4. Normalizing text according to the `Italian Twitter word embedding <http://www.italianlp.it/download-italian-twitter-embeddings>`_
    5. Substitute special entities (hashtags, emoji, ...)
    6. Remove punctuation (again)
    7. Tokenization based on italian language (e.g. l\'albero -> l\', albero)

    :param tweets:
    :return:
    """

    m = tweets

    # filter
    # max_words = int(pd.Series(map(lambda x: len(x.split()), m)).quantile(MAX_NWORDS_QUANTILE))
    m = filter(lambda x: len(x.split()) <= SEQ_LENGTH, m)

    # lowercase
    m = map(str.lower, m)

    # remove punct
    # punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    punct = """!"#$%&()*+-;<=>?[\]^`{|}~"""
    table = str.maketrans('', '', punct)
    m = map(lambda x: x.translate(table), m)

    # normalize_text
    # m = map(lambda x: normalize_text(x.split()), m)
    # m = map(normalize_text, m)
    m = map(lambda sentence: ' '.join([normalize_text(word) for word in sentence.split()]), m)

    # tweet-preprocessor
    m = map(tweet_parsing, m)

    # remove punct (2)
    punct2 = ':.,@'  # _
    table2 = str.maketrans('', '', punct2)
    m = map(lambda x: x.translate(table2), m)

    # tokenize
    # m = map(tokenize, m)
    # t.tokenize(sentence)

    # encode

    m, m2 = itertools.tee(m)
    t = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    tokenizer = Tokenizer(t)
    list(map(lambda x: tokenizer.fit_on_texts(t.tokenize(x)), m))
    m = map(lambda x: [START_TOKEN] + [tokenizer.word_index.get(w) for w in t.tokenize(x) if tokenizer.word_index.get(w)], m2)

    return m, tokenizer


def normalize_text(word):
    """
    Numbers:
    - Integer numbers between 0 and 2100 were kept as original
    - Each integer number greater than 2100 is mapped in a string which represents the number of digits needed to store the number (ex: 10000 \-> DIGLEN\_5)
    - Each digit in a string that is not convertible to a number must be converted with the following char: @Dg. This is an example of replacement (ex: 10,234 \-> @Dg@Dg,@Dg@Dg@Dg)

    Words:
    - A string starting with lower case character must be lowercased
      (e.g.: (“aNtoNio” -> “antonio”), (“cane” -> “cane”))
    - A string starting with an upcased character must be capitalized
      (e.g.: (“CANE” -> “Cane”, “Antonio” -> “Antonio”))

    :param word:
    :return:
    """
    if "http" in word or ("." in word and "/" in word):
        word = unicode("___URL___")
        # word = '\\URL'
        return word
    if len(word) > 26:
        return "__LONG-LONG__"
        # return "\\LONGLONG"
    new_word = _get_digits(word)
    if new_word != word:
        word = new_word
    if word[0].isupper():
        word = word.capitalize()
    else:
        word = word.lower()
    return word


def _get_digits(text):
    """
    1.  Integer numbers between 0 and 2100 were kept as original
    2.  Each integer number greater than 2100 is mapped in a string which represents the number of digits needed to
        store the number (e.g.: 10000 -> DIGLEN_5)
    3.  Each digit in a string that is not convertible to a number must be converted with the following char: @Dg.
        This is an example of replacement (ex: 10,234 -> @Dg@Dg,@Dg@Dg@Dg)

    :param text:
    :return:
    """
    try:
        val = int(text)
    except:
        text = re.sub('\d', '@Dg', text)
        # text = re.sub('\d', '\\\Dg', text)
        return text
    if val >= 0 and val < 2100:
        return str(val)
    else:
        return "DIGLEN_" + str(len(str(val)))
        # return "\\DIGLEN" + str(len(str(val)))


def tokenize(sentence):
    # try:
    #     tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    # except LookupError:
    #     nltk.download('punkt')
    #     tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    t = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    return t.tokenize(sentence)


# def filter_tweets(tweets):
#     # remove tweets with too many words
#     tweets['full_text_nwords'] = tweets.apply(lambda x: len(x['full_text'].split()), axis=1)
#     max_words = int(tweets['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
#     return list(tweets[tweets['full_text_nwords'] <= max_words]['full_text']), max_words
#
#
# def encode_tweets(tweets):
#
#     # encode
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(tweets)
#     encoded_tweets = tokenizer.texts_to_sequences(tweets)
#
#     # # pad documents
#     # proc_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_words, padding='post')
#
#     encoded_tweets = [[START_TOKEN] + tweet for tweet in encoded_tweets]
#
#     return encoded_tweets, tokenizer
#

EMOJI_PREFIX = '/EMOJI'
HASHTAG_PREFIX = '/HASHTAG'
URL_PREFIX = '/URL'
MENTIONS_PREFIX = '/MENTION'


def tweet_parsing(tweet):
    """

    :param tweet:
    :return:
    """

    # create the parser
    parser_tweet = preprocessor.parse(tweet)

    # check and substitute for the element we do not want
    if parser_tweet.emojis:
        for emoji in parser_tweet.emojis:
            emoji_str = emoji.match
            tweet = tweet.replace(emoji_str, '')

    if parser_tweet.hashtags:
        for hashtag in parser_tweet.hashtags:
            hashtag_str = hashtag.match
            tweet = tweet.replace(hashtag_str, hashtag_str[1:])

    # if parser_tweet.urls:
    #     for url in parser_tweet.urls:
    #         url_str = url.match
    #         tweet = tweet.replace(url_str, URL_PREFIX)

    if parser_tweet.mentions:
        for mention in parser_tweet.mentions:
            mention_str = mention.match
            tweet = tweet.replace(mention_str, '')

    # Preprocessor is /HASHTAG#awesome 👍 /URLhttps://github.com/s/preprocessor
    return tweet


if __name__ == '__main__':
    print(_get_digits("15"))
    print(_get_digits("2099"))
    print(_get_digits("2100"))
    print(_get_digits("Ho visto 3000 manigoldi"))
    print(normalize_text("aNtoNio"))
    print(normalize_text("cane"))
    print(normalize_text("CANE"))
    print(normalize_text("Ho visto 3000 manigoldi"))
    print(normalize_text("100,23"))
    s = "Questo è l'apostrofo, con una virgola ed un punto e virgola;"
    t = nltk.data.load('tokenizers/punkt/italian.pickle')
    print(t.tokenize(s))
    t2 = nltk.RegexpTokenizer(r"[dnl]['´`]|\w+|\$[\d\.]+|\S+")
    print(t2.tokenize(s))
