
import logging

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.callbacks import ModelCheckpoint
from os.path import join, dirname

INPUT_DIR = join(dirname(__file__), 'shared')
WORD_EMBEDDING_PATH = join(INPUT_DIR, 'twitter128.sqlite')
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
    # https://keras.io/guides/understanding_masking_and_padding/
    # Recurrent layer
    model.add(LSTM(256))
    # Dropout for regularization
    model.add(Dropout(0.2))
    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [checkpoint]

    # fit the model
    model.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=callbacks,
              validation_data=(X_valid, y_valid))

    return model


