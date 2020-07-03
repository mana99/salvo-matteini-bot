
import pandas as pd
import pickle

from os.path import join, dirname
from model import partion_dataset, train_validate_model, test_model

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/
from preprocessing import preprocess

INPUT_DIR = join(dirname(__file__), 'shared')
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
TOKENIZER_PATH = join(INPUT_DIR, 'tokenizer.pickle')


# import
tweets_df = pd.read_csv(INPUT_PATH, usecols=['full_text']) #[:100]
# for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNKSIZE, usecols=['full_text']):
#      print(chunk)

# todo preprocessing
processed_tweets, t, vocab_size, max_words = preprocess(tweets_df)
with open(TOKENIZER_PATH, 'wb') as fout:
    pickle.dump(t, fout)

# todo partitioning
splitted_data = partion_dataset(input_data=processed_tweets)

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
