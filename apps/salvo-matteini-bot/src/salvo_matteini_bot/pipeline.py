import logging
import pickle

import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from salvo_matteini_bot import INPUT_PATH, TOKENIZER_PATH, PREPROCESSED_TWEETS_PATH, \
    TRAIN_VALID_TEST_RATIO, SEQ_LENGTH, SPLITTED_DATASETS_PATH, NUM_EPOCHS, MODEL_PATH, SCORE_PATH, \
    EMBEDDING_MATRIX_PATH
from salvo_matteini_bot.data_import import import_data, mongo_import
from salvo_matteini_bot.embedding import get_t128_italiannlp_embedding
from salvo_matteini_bot.labeling import get_features_labels
from salvo_matteini_bot.model import compile_model
from salvo_matteini_bot.partitioning import partition_tweets
from salvo_matteini_bot.preprocessing import preprocess


logger = logging.getLogger(__name__)


def execute(start: int = 0, export: bool = False):

    # 0. import
    raw_tweets = _import(cached=start>0)
    # 1. pre-process
    tokenizer, tweets = _preprocess(raw_tweets, cached=start>1, export=export)
    # 2. embedding
    embedding_matrix = _embedding(tokenizer, cached=start>2, export=export)
    # 3. partition + label
    Xy = _partition(tweets, embedding_matrix, cached=start>3, export=export)
    # 4. model train + validate
    model = _model_train_validate(Xy["train"], Xy["validation"], embedding_matrix, cached=start>4, export=export)
    # 5. model test
    metrics_values = _model_test(Xy["test"], model)
    # 6. model score
    score = _model_score(Xy["score"], model, tokenizer, embedding_matrix)



# # todo cache + export decorator
# def import_export(func):
#     def wrap_func(*args, **kwargs):
#         fn = wrap_func.__name__
#         if cached:
#             with open(fn, 'rb') as fin:
#                 output = pickle.load(fin)
#             return output
#         else:
#             output = func(*args, **kwargs) * 10
#             if export:
#                 with open(fn, 'wb') as fout:
#                     pickle.dump(output, fout)
#             return output
#     return wrap_func


def _import(cached=False):
    logger.info("Importing")
    if cached:
        return
    else:
        tweets = mongo_import()
        return tweets


def _preprocess(tweets, cached=False, export=False):
    logger.info("Preprocessing")
    if cached:
        with open(TOKENIZER_PATH, 'rb') as fin:
            tokenizer = pickle.load(fin)
        with open(PREPROCESSED_TWEETS_PATH, 'rb') as fin:
            tweets = pickle.load(fin)
        return tokenizer, tweets
    else:
        tweets_iterator, tokenizer = preprocess(tweets)
        tweets = list(tweets_iterator)
        if export:
            logger.debug("Exporting tokenizer and pre-processed tweets")
            with open(TOKENIZER_PATH, 'wb') as fout:
                pickle.dump(tokenizer, fout)
            with open(PREPROCESSED_TWEETS_PATH, 'wb') as fout:
                pickle.dump(tweets, fout)
        return tokenizer, tweets


def _embedding(tokenizer, cached=False, export=False):
    logger.info("Creating Embedding Matrix")
    if cached:
        with open(EMBEDDING_MATRIX_PATH, 'rb') as fin:
            embedding_matrix = pickle.load(fin)
        return embedding_matrix
    else:
        embedding_matrix = get_t128_italiannlp_embedding(tokenizer=tokenizer)
        if export:
            logger.debug("Exporting embedding matrix")
            with open(EMBEDDING_MATRIX_PATH, 'wb') as fout:
                pickle.dump(embedding_matrix, fout)
        return embedding_matrix


def _partition(tweets, embedding_matrix, cached=False, export=False):
    # todo dict
    if cached:
        with open(SPLITTED_DATASETS_PATH, 'rb') as fin:
            Xys = pickle.load(fin)
        return Xys
    else:

        logger.info("Partitioning")
        partitioned_tweets = partition_tweets(tweets, TRAIN_VALID_TEST_RATIO)

        logger.info("Generating labels")
        X_train, y_train = get_features_labels(partitioned_tweets["train"], SEQ_LENGTH, embedding_matrix)
        X_valid, y_valid = get_features_labels(partitioned_tweets["validation"], SEQ_LENGTH, embedding_matrix)
        X_test, y_test = get_features_labels(partitioned_tweets["test"], SEQ_LENGTH, embedding_matrix)

        output = {
            "train": {
                "X": X_train,
                "y": y_train
            },
            "validation": {
                "X": X_valid,
                "y": y_valid
            },
            "test": {
                "X": X_test,
                "y": y_test
            },
            "score": {
                "X": partitioned_tweets["score"]
            }
        }

        if export:
            with open(SPLITTED_DATASETS_PATH, 'wb') as fout:
                pickle.dump(output, fout)

        return output


def _model_train_validate(train_data, valid_data, embedding_matrix, cached=False, export=False):
    if cached:
        model = load_model(MODEL_PATH)
        model.summary()
        return model
    else:

        logger.info("Build model")
        model, callbacks = compile_model(train_data["X"], train_data["y"], embedding_matrix=embedding_matrix)
        model.summary()

        logger.info("Train and validate model")
        model.fit(train_data["X"], train_data["y"], epochs=NUM_EPOCHS, batch_size=128, callbacks=callbacks,
                  validation_data=(valid_data["X"], valid_data["y"]))

        if export:
            logger.info("Saving final model to file")
            model.save(MODEL_PATH)

        return model


def _model_test(test_data, model):
    # Evaluate on validation data
    metrics_values = model.evaluate(test_data["X"], test_data["y"], verbose=0)
    if len(model.metrics_names) == 1:
        print("%s: %.2f%%" % (model.metrics_names[0], metrics_values*100))
    else:
        print(', '.join(["%s: %.2f%%" % (x, y) for x, y in zip(model.metrics_names, metrics_values)]))
    return metrics_values


def _model_score(score_data, model, tokenizer, embedding_matrix, ):
    # todo: optimize
    X_score = [word
               for tweet in score_data["X"]
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
