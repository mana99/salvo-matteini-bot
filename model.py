
import logging

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.callbacks import ModelCheckpoint
from data_preparation_for_the_net import *
from os.path import join, dirname

INPUT_DIR = join(dirname(__file__), 'shared')
WORD_EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.sqlite')
WORD_EMBEDDING_SIZE = 128
MODEL_PATH = join(INPUT_DIR, 'model.h5')

logger = logging.getLogger(__name__)


# X.shape: (n_examples, n_batch, 1)


def train_validate_model(X_train, y_train, X_valid, y_valid, *, embedding_matrix):

    # print("train len {}, validation len {}".format(len(X_train), len(X_valid)))

    model = Sequential()
    # Embedding layer
    # we do want to update the learned word weights in this model, therefore we will set the trainable attribute for
    # the model to be True.
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=X_train.shape[1],
                        weights=[embedding_matrix],
                        trainable=True,
                        mask_zero=True))
    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))
    # Recurrent layer
    model.add(LSTM(256))
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
