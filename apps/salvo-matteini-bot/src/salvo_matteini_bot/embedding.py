
import logging
import ujson
import numpy as np

from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from salvo_matteini_bot import EMBEDDING_PATH

logger = logging.getLogger(__name__)


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/
def get_t128_italiannlp_embedding(tokenizer: Tokenizer) -> np.array:
    """
    Load the 128-sized italian word embedding trained on tweets by the Italian Natural Language Processing  Lab [1]
    (see http://www.italianlp.it/resources/italian-word-embeddings/) and build the embedding matrix for the
    the current tokenization.

    ---

    [1] Cimino A., De Mattei L., Dell’Orletta F. (2018) "`Multi-task Learning in Deep Neural Networks at EVALITA 2018
    <http://ceur-ws.org/Vol-2263/paper013.pdf>`_". In Proceedings of EVALITA ’18, Evaluation of NLP and Speech Tools
    for Italian, 12-13 December, Turin, Italy.

    :param tokenizer: the tokenizer object
    :return: the (n_words x 128) embedding matrix
    """

    # t128 size: 1188949, 1027699 (lowercase)

    n_words = len(tokenizer.word_index) + 1

    # create a weight matrix for words in training docs
    # initialize as random and not to zeros to avoid cosine similarity issues
    embedding_matrix = np.random.uniform(low=-1, high=1, size=(n_words, 128))
    # todo: constant size

    # load the whole embedding into memory
    # takes a while... (~1-2 min)
    logger.info("Loading pre-trained word embedding in memory (~1-2 mins)...")
    with open(EMBEDDING_PATH, 'r') as fin:
        t128 = ujson.load(fin)

    logger.info("Building embedding matrix...")
    for word, i in tqdm(tokenizer.word_index.items()):
        index = t128.get(word)
        if index:
            embedding_matrix[i] = index[:-1]

    # sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
    # connection = sql_engine.raw_connection()
    # for word, i in tokenizer.word_index.items():
    #     res = t128[t128['key_lower'] == word.lower()]  # troppo lento
    #     res = pd.read_sql(sql=f'select * from store where key = "{word}"', con=connection)
    #     if len(res) == 1:
    #         embedding_matrix[i] = res.drop(['key', 'ranking'], axis=1).values[0]

    return embedding_matrix