
import logging

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sqlalchemy import create_engine
from data_preparation_for_the_net import *
from sklearn.utils import shuffle
from os.path import join, dirname


INPUT_DIR = join(dirname(__file__), 'shared')
WORD_EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.sqlite')
WORD_EMBEDDING_SIZE = 128
MODEL_PATH = join(INPUT_DIR, 'model.h5')

logger = logging.getLogger(__name__)


def partion_dataset(features, labels, train_perc=0.7, random_state=None, training_length=10):

    # aggregator = cleaning()
    # # features = features_labels_with_strings(aggregator, training_length)
    # # fixed_length_shorter_makes_sense = 998 # 500
    # # aggregator = aggregator[0:fixed_length_shorter_makes_sense]
    #
    # aggregator_integers = integers_conversion(aggregator)
    # features, labels = features_labels(aggregator_integers,
    #                                    training_length)  # check if the features should be like these
    # print(features, labels)
    #
    # label_array = create_one_hot_encoding(features, labels)

    # it mantains the same shuffling order for both arrays
    shuffled_features, shuffled_labels = shuffle(features, labels, random_state=random_state)

    train_index = int(len(shuffled_features) * train_perc)
    X_train = shuffled_features[:train_index]
    y_train = shuffled_labels[:train_index]
    X_valid = shuffled_features[train_index:]
    y_valid = shuffled_labels[train_index:]

    return X_train, y_train, X_valid, y_valid


# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# http://www.italianlp.it/resources/italian-word-embeddings/

def get_t128_italiannlp_embedding(tokenizer, vocab_size, max_words):

    # t128 size: 1188949, 1027699 (lower)

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDING_SIZE))

    # load the whole embedding into memory
    # takes a while... (~1-2 min)
    logger.info("Loading pre-trained word embedding in memory (~1-2 mins)...")
    with open(join(INPUT_DIR, 'twitter128.json'), 'r') as fin:
        t128 = json.load(fin)

    logger.info("Building embedding matrix...")
    for word, i in tokenizer.word_index.items():
        embedding_matrix[i] = t128.get(word, list(np.random.choice([1, -1]) * np.random.rand(WORD_EMBEDDING_SIZE+1)))[:-1]

    # sql_engine = create_engine(f"sqlite:///{WORD_EMBEDDING_PATH}")
    # connection = sql_engine.raw_connection()
    # for word, i in tokenizer.word_index.items():
    #     res = t128[t128['key_lower'] == word.lower()]  # troppo lento
    #     res = pd.read_sql(sql=f'select * from store where key = "{word}"', con=connection)
    #     if len(res) == 1:
    #         embedding_matrix[i] = res.drop(['key', 'ranking'], axis=1).values[0]

    return embedding_matrix


def train_validate_model(X_train, y_train, X_valid, y_valid, *, tokenizer, vocab_size, max_words):

    print("train len {}, validation len {}".format(len(X_train), len(X_valid)))
    model = Sequential()


    # embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer,
    #                                                  vocab_size=vocab_size,
    #                                                  max_words=max_words)
    #
    # model.add(Embedding(input_dim=(X_train.shape[1], X_train.shape[2]),
    #                     output_dim=WORD_EMBEDDING_SIZE,
    #                     input_length=max_words,
    #                     weights=[embedding_matrix],
    #                     trainable=True,
    #                     mask_zero=True))
    # # we do want to update the learned word weights in this model, therefore we will set the trainable attribute for
    # # the model to be True.


    # Recurrent layer
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Dropout for regularization
    model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=callbacks_list)
#              validation_data=(X_valid, y_valid))


    # Embedding layer
    # model.add(get_t128_italiannlp_embedding(tokenizer=tokenizer,
    #                                         vocab_size=vocab_size,
    #                                         max_words=max_words))
    # # Masking layer for pre-trained embeddings
    # model.add(Masking(mask_value=0.0))
    # # Recurrent layer
    # model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    # # Fully connected layer
    # model.add(Dense(64, activation='relu'))
    # # Dropout for regularization
    # model.add(Dropout(0.5))
    # # Output layer
    # model.add(Dense(vocab_size+1, activation='softmax'))
    # todo restituire classificatore o embedding layer
    # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #
    # # Create callbacks
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # model_checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, save_weights_only=False)
    # callbacks = [early_stopping, model_checkpoint]
    #
    # logger.info("Fitting model...")
    # model.fit(X_train, y_train,
    #                     batch_size=2048, epochs=150,
    #                     callbacks=callbacks,
    #                     validation_data=(X_valid, y_valid))

    return model


def test_model(loaded_model, start_word):

    # loaded_model.load_weights('models/model.h5')

    to_pred = np.array([start_word])
    predicted_result = loaded_model.predict(to_pred)
    print("predicted result is {}".format(predicted_result))
    input_phrase = ''

    ###########################
    # prediction not integers #
    """
    for pred in predicted_result[0]:
        input_phrase += seq_dict[pred]
    print(input_phrase)
    """
