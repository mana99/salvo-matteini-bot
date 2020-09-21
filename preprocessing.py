
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from os.path import join, dirname

MAX_NWORDS_QUANTILE = 0.99
INPUT_DIR = join(dirname(__file__), 'shared')
TOKENIZER_PATH = join(INPUT_DIR, 'tokenizer.pickle')

START_TOKEN, END_TOKEN = 0, -1
# non usare -1 (errore in embedding layer)


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

    # # pad documents
    # proc_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_words, padding='post')

    proc_encoded_tweets = [[START_TOKEN] + tweet for tweet in encoded_tweets]

    return proc_encoded_tweets, t, vocab_size, max_words

