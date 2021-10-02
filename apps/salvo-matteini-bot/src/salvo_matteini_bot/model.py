
import logging
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.losses_utils import ReductionV2

logger = logging.getLogger(__name__)


def euclidean_distance(y_true, y_pred):
    # https://www.tutorialexample.com/calculate-euclidean-distance-in-tensorflow-a-step-guide-tensorflow-tutorial/
    # return tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(y_true - y_pred))))
    return tf.reduce_mean(tf.norm(y_true - y_pred))


def cosine_similarity(a, b):
    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity
    # return tf.losses.CosineSimilarity(axis=-1, reduction=ReductionV2.AUTO)(a, b)
    return tf.losses.CosineSimilarity(a, b)


# X.shape: (n_examples, n_batch, 1)
def compile_model(X_train, y_train, *, embedding_matrix):

    # print("train len {}, validation len {}".format(len(X_train), len(X_valid)))

    model = Sequential()
    # Embedding layer
    # we do NOT want to update the learned word weights in this model,
    # therefore we will set the trainable attribute for the model to be False.
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=X_train.shape[1],
                        weights=[embedding_matrix],
                        trainable=False,
                        mask_zero=False))
    # todo: embedding layer trainable and return
    # # Masking layer for pre-trained embeddings
    # model.add(Masking(mask_value=0.0))
    # https://keras.io/guides/understanding_masking_and_padding/
    # Recurrent layer
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))

    # # Dropout for regularization
    # model.add(Dropout(0.2))

    # Dense layer
    model.add(Dense(256, activation='relu'))

    # Output layer
    model.add(Dense(y_train.shape[1], activation='softmax'))

    # We want to use cosine similarity as it is used in word2vec which is used to build the twitter128 embedding
    model.compile(loss=tf.losses.CosineSimilarity(reduction=ReductionV2.AUTO), optimizer='adam')
    # model.compile(loss=tf.losses.CosineSimilarity(axis=-1, reduction=ReductionV2.AUTO), optimizer='adam')
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # define the checkpoint
    filepath = "/tmp/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [checkpoint]

    return model, callbacks
