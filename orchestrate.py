
import numpy as np
import pandas as pd

from os.path import join, dirname
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from model import obtain_train_validation_dataset, train_validate_model, model_test

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/


INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
CHUNKSIZE = 10 ** 6
VOCAB_SIZE = 2000
MAX_NWORDS_QUANTILE = 0.99


# for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
#      print(chunk)

tweets_df = pd.read_csv(INPUT_PATH, usecols=['full_text']) #[:100]

# todo preprocessing

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

# todo partitioning
splitted_data = obtain_train_validation_dataset(input_data=padded_encoded_tweets)

# train and validate model
model = train_validate_model(splitted_data=splitted_data,
                             tokenizer=t,
                             vocab_size=vocab_size,
                             max_words=max_words)

# save model
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("models/model.h5")

# todo evaluate
# print()
# Load in model and evaluate on validation data
# model = tf.keras.models.load_model('../models/model.h5')  # load_model('../models/model.h5')
# model.evaluate(X_valid, y_valid)


# todo score model
# score_output = model_test(score_input)
