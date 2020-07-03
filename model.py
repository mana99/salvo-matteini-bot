from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.models import model_from_json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models
import tensorflow as tf

from data_preparation_for_the_net import *

from sklearn.utils import shuffle

"""with open("models/model.h5", "w") as json_file:
    json_file.write('')"""


def obtain_train_validation_dataset():
    aggregator = cleaning()
    train_perc = 0.7
    training_length = 10
    # features = features_labels_with_strings(aggregator, training_length)
    # fixed_length_shorter_makes_sense = 998 # 500
    # aggregator = aggregator[0:fixed_length_shorter_makes_sense]

    aggregator_integers = integers_conversion(aggregator)
    features, labels = features_labels(aggregator_integers,
                                       training_length)  # check if the features should be like these
    print(features, labels)

    label_array = create_one_hot_encoding(features, labels)

    # it mantains the same shuffling order for both arrays
    shuffled_features, shuffled_labels = shuffle(features, labels, random_state=0)

    train_index = int(len(shuffled_features) * train_perc)
    X_train = shuffled_features[:train_index]
    y_train = shuffled_labels[:train_index]
    X_valid = shuffled_features[train_index:]
    y_valid = shuffled_labels[train_index:]

    return X_train, y_train, X_valid, y_valid


def model_creation(X_train, y_train, X_valid, y_valid):

    training_length = 10
    num_words = len(X_train) + 1
    print("train len {}, validation len {}".format(num_words, num_words + len(X_valid)))
    model = Sequential()
    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                  input_length=training_length,
                  output_dim=100,
                  # weights=[embedding_matrix],
                  trainable=False,
                  mask_zero=True))
    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))
    # Recurrent layer
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    # Fully connected layer
    model.add(Dense(64, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(num_words, activation='softmax'))
    # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint('models/model.h5', save_best_only=True, save_weights_only=False)
    callbacks = [early_stopping, model_checkpoint]

    history = model.fit(X_train, y_train,
                        batch_size=2048, epochs=150,
                        callbacks=callbacks,
                        validation_data=(X_valid, y_valid))
    print("history is : \n{}".format(history))

    ####### SAVE MODEL #########
    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("models/model.h5")
    print("Saved model to disk")
    ############################

    # print()
    # Load in model and evaluate on validation data
    # model = tf.keras.models.load_model('../models/model.h5')  # load_model('../models/model.h5')
    model.evaluate(X_valid, y_valid)


def model_test():
    print("test")
    with open("models/model.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('models/model.h5')

    to_pred = [14, 64, 16, 65, 41, 66, 44, 67, 68, 69]

    seq_dict = read_dict(seq_dict_path)
    input_phrase = ''
    for pred in to_pred:
        input_phrase += seq_dict[pred] + ' '

    print(input_phrase)

    to_pred = np.array([to_pred])
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


def main():
    model = model_creation()
    model_test()


if __name__ == '__main__':
    main()
