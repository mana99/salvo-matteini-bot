
from os.path import join, dirname

# todo env variables
RANDOM_STATE = 1337

PRJ_DIR = join(join(dirname(__file__), '..'), '..')

INPUT_DIR = join(PRJ_DIR, "input")
OUTPUT_DIR = join(PRJ_DIR, "output")

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


# preprocessing
SEQ_LENGTH = 52
# MAX_NWORDS_QUANTILE = 0.99
START_TOKEN = 0  # non usare -1 (errore in embedding layer)

# partitioning
TRAIN_VALID_TEST_RATIO = (0.55, 0.3, 0.15)

# modeling
NUM_EPOCHS = 5

