
import pandas as pd
import tensorflow as tf
import logging
import pickle

from tensorflow.python.keras.utils.losses_utils import ReductionV2

from salvo_matteini_bot.data_import import import_data
from salvo_matteini_bot.preprocessing import preprocess
from salvo_matteini_bot.embedding import get_t128_italiannlp_embedding
from salvo_matteini_bot.partitioning import partition_tweets
from salvo_matteini_bot.labeling import get_features_labels
from salvo_matteini_bot.model import compile_model

from salvo_matteini_bot import (
    INPUT_PATH,
    MODEL_PATH,
    EMBEDDING_PATH,
    PREPROCESSED_TWEETS_PATH,
    TOKENIZER_PATH,
    SCORE_PATH,
    SPLITTED_DATASETS_PATH,
    SEQ_LENGTH,
    NUM_EPOCHS,
    TRAIN_VALID_TEST_RATIO
)

logger = logging.getLogger(__name__)


def main():

    # todo argparse
    # todo starting point parameter

    # with open(TOKENIZER_PATH, 'rb') as fin:
    #     tokenizer = pickle.load(fin)
    # with open(PREPROCESSED_TWEETS_PATH, 'rb') as fin:
    #     tweets = pickle.load(fin)
    # with open(EMBEDDING_MATRIX_PATH, 'rb') as fin:
    #     embedding_matrix = pickle.load(fin)
    # with open(SPLITTED_DATASETS_PATH, 'rb') as fin:
    #     X_train, y_train, X_valid, y_valid, X_test, y_test, tweets_score = pickle.load(fin)
    # model = load_model(MODEL_PATH)

    # Import
    print("Importing")
    tweets = import_data(INPUT_PATH)

    # Preprocessing
    print("Preprocessing")
    tweets_iterator, tokenizer = preprocess(tweets['full_text'])
    with open(TOKENIZER_PATH, 'wb') as fout:
        pickle.dump(tokenizer, fout)
    vocab_size = len(tokenizer.word_index) + 1
    tweets = list(tweets_iterator)
    with open(PREPROCESSED_TWEETS_PATH, 'wb') as fout:
        pickle.dump(tweets, fout)

    # Create embedding matrix
    print("Creating Embedding Matrix")
    embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer, n_words=vocab_size)
    with open(EMBEDDING_PATH, 'wb') as fout:
        pickle.dump(embedding_matrix, fout)

    # Partitioning
    print("Partitioning")
    tweets_train, tweets_valid, tweets_test, tweets_score = partition_tweets(tweets, TRAIN_VALID_TEST_RATIO)

    # Get labels
    print("Generating labels")
    X_train, y_train = get_features_labels(tweets_train, vocab_size, SEQ_LENGTH, embedding_matrix)
    X_valid, y_valid = get_features_labels(tweets_valid, vocab_size, SEQ_LENGTH, embedding_matrix)
    X_test, y_test = get_features_labels(tweets_test, vocab_size, SEQ_LENGTH, embedding_matrix)

    with open(SPLITTED_DATASETS_PATH, 'wb') as fout:
        pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test, tweets_score), fout)

    # Compile model
    print("Build model")
    model, callbacks = compile_model(X_train, y_train, embedding_matrix=embedding_matrix)

    # # Train and validate model
    print("Train and validate model")
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=128, callbacks=callbacks,
              validation_data=(X_valid, y_valid))

    # # Save model
    print("Saving final model to file")
    model.save(MODEL_PATH)

    # Evaluate on validation data
    metrics_values = model.evaluate(X_test, y_test, verbose=0)
    if len(model.metrics_names) == 1:
        print("%s: %.2f%%" % (model.metrics_names[0], metrics_values*100))
    else:
        print(', '.join(["%s: %.2f%%" % (x, y) for x, y in zip(model.metrics_names, metrics_values)]))

    # Score model
    # todo: optimize
    X_score = [word
               for tweet in tweets_score
               for word in tweet][:SEQ_LENGTH]

    def get_word(index):
        if index == 0:
            return '\\START'
        else:
            return tokenizer.index_word[index]

    score_words = [get_word(x) for x in X_score]
    embedding_matrix_float32 = tf.cast(embedding_matrix, float).numpy()
    embedding_matrix_df = pd.DataFrame(embedding_matrix_float32)
    dist = lambda x: tf.losses.CosineSimilarity(reduction=ReductionV2.AUTO)(x, predicted_result[-1]).numpy()
    MAX = 30
    for i in range(1, MAX):
        print(f"Predicting word {i}/{MAX}...")
        # predict
        predicted_result = model.predict(X_score)
        # nearest word in embedding vector
        dist_df = embedding_matrix_df.apply(dist, axis=1)
        predicted_embedding = abs(dist_df).argmin()
        # index to word
        predicted_word = get_word(predicted_embedding)
        print(predicted_word)
        score_words.append(predicted_word)
        # add predicted word to the input
        X_score = X_score[1:] + [predicted_embedding]

    with open(SCORE_PATH, 'w') as fout:
        fout.write(" ".join(score_words))

    # todo: closure implementation
    # def predict_sentence():
    #     sentence = []
    #
    #     def predictor(sequence=None):
    #         nonlocal sentence
    #         if not sequence and not sentence:
    #             raise Exception
    #         elif sequence:
    #             if sentence:
    #                 print('Restarting')
    #             sentence = sequence
    #             prediction = model.predict(sequence)
    #             sentence.append(prediction)
    #         elif sentence:
    #             prediction = model.predict(sentence[-4:])
    #             sentence.append(prediction)
    #         return sentence
    #
    #     return predictor
    #
    # predict_sentence(X_score)
    # predict_sentence()
    # predict_sentence()
    # predict_sentence()
    # sentence = predict_sentence()


if __name__ == '__main__':
    main()
