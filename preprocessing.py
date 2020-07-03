
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os.path import join, dirname

MAX_NWORDS_QUANTILE = 0.99
INPUT_DIR = join(dirname(__file__), 'shared')
TOKENIZER_PATH = join(INPUT_DIR, 'tokenizer.pickle')


def preprocess(tweets_df):

    # todo remove characters

    # todo entities (e.g. https://www.debian.org/ --> \URL) (see utils.tweet_parsing)

    # remove tweets with too many words
    tweets_df['full_text_nwords'] = tweets_df.apply(lambda x: len(x['full_text'].split()), axis=1)
    max_words = int(tweets_df['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
    tweets = list(tweets_df[tweets_df['full_text_nwords'] <= max_words]['full_text'])

    # encode
    t = Tokenizer()
    t.fit_on_texts(tweets)
    vocab_size = len(t.word_index) + 1   # todo: constant
    encoded_tweets = t.texts_to_sequences(tweets)

    # pad documents
    padded_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_words, padding='post')

    return padded_encoded_tweets, t, vocab_size, max_words
