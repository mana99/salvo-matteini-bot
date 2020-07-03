
from os.path import join, dirname
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from rnn_model import obtain_train_validation_dataset, model_creation, model_test

INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
CHUNKSIZE = 10 ** 6
VOCAB_SIZE = 2000
MAX_NWORDS_QUANTILE = 0.99


# for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
#      print(chunk)

tweets_df = pd.read_csv(INPUT_PATH, usecols=['full_text'])

# todo preprocessing

# remove tweets with too many words
tweets_df['full_text_nwords'] = tweets_df.apply(lambda x: len(x['full_text'].split()), axis=1)
max_length = int(tweets_df['full_text_nwords'].quantile(MAX_NWORDS_QUANTILE))
tweets = list(tweets_df[tweets_df['full_text_nwords'] <= max_length]['full_text'])

# encode
t = Tokenizer()
t.fit_on_texts(tweets)
vocab_size = len(t.word_index) + 1   # todo: constant
encoded_tweets = t.texts_to_sequences(tweets)

# pad documents
padded_encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding='post')

# todo partitioning
# X_train, y_train, X_valid, y_valid = obtain_train_validation_dataset(padded_encoded_tweets)

# todo train and validate model
# model = model_creation(X_train, y_train)

# todo score model
# score_output = model_test(score_input)
