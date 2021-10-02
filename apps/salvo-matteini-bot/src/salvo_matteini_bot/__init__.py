
from os import environ
from os.path import join, dirname, abspath
from functools import reduce

MONGO_URL = environ["MONGO_URL"]
MONGO_DB_NAME = environ["MONGO_DB_NAME"]
MONGO_COLL_NAME = environ["MONGO_COLL_NAME"]

# PRJ_DIR = abspath(reduce(join, [dirname(__file__), "..", "..", "..", ".."]))

# INPUT_DIR = join(PRJ_DIR, "input")
# OUTPUT_DIR = join(PRJ_DIR, "output")
INPUT_DIR = environ["INPUT_DIR"]
OUTPUT_DIR = environ["OUTPUT_DIR"]

# input path
INPUT_PATH = join(INPUT_DIR, 'tweets.csv')
# embedding path
EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.json')
# partial results paths
TOKENIZER_PATH = join(OUTPUT_DIR, 'tokenizer.pickle')
PREPROCESSED_TWEETS_PATH = join(OUTPUT_DIR, 'preprocessed-tweets.pickle')
EMBEDDING_MATRIX_PATH = join(OUTPUT_DIR, 'embedding.pickle')
SPLITTED_DATASETS_PATH = join(OUTPUT_DIR, 'splitted_datasets.pickle')
# saved model path
MODEL_PATH = join(OUTPUT_DIR, 'model.h5')
# score path
SCORE_PATH = join(INPUT_DIR, 'score.txt')

# todo configuration file
# preprocessing
RANDOM_STATE = 1337
SEQ_LENGTH = 52
# MAX_NWORDS_QUANTILE = 0.99

# partitioning
TRAIN_VALID_TEST_RATIO = (0.55, 0.3, 0.15)

# modeling
NUM_EPOCHS = 5
