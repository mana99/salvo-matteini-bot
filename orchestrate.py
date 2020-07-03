
import numpy as np
import pandas as pd

from os.path import join, dirname
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from sqlalchemy import create_engine

from model import obtain_train_validation_dataset, model_creation, model_test

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/


INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
WORD_EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.sqlite')
WORD_EMBEDDING_SIZE = 128
CHUNKSIZE = 10 ** 6
VOCAB_SIZE = 2000
MAX_NWORDS_QUANTILE = 0.99


# for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
#      print(chunk)

tweets_df = pd.read_csv(INPUT_PATH, usecols=['full_text']) #[:100]

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

# load the whole embedding into memory
sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
connection = sql_engine.raw_connection()
t128 = pd.read_sql(sql='select * from store --limit 100', con=connection)
# todo filter words in t.word_index
t128['key_lower'] = t128['key'].apply(str.lower)

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDING_SIZE))
for word, i in t.word_index.items():
    res = t128[t128['key_lower'] == word.lower()]
    if len(res) == 1:
        embedding_matrix[i] = res.drop(['key', 'key_lower', 'ranking'], axis=1).values[0]

# we do want to update the learned word weights in this model, therefore we will set the trainable attribute for
# the model to be True.
e = Embedding(input_dim=vocab_size, output_dim=WORD_EMBEDDING_SIZE, input_length=max_length,
              weights=[embedding_matrix], trainable=True, mask_zero=True)

# todo partitioning
# X_train, y_train, X_valid, y_valid = obtain_train_validation_dataset(padded_encoded_tweets)

# todo train and validate model
# model = model_creation(X_train, y_train)

# todo score model
# score_output = model_test(score_input)
