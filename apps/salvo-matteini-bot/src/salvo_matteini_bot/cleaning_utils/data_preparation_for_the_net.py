# import keras
import re
import os
import json
import numpy as np
import pandas as pd

pattern_list = ['Inferno: Canto.*', 'INFERNO']
DIRECTORY = 'tweets'
file_name = 'tweets_csv_test.csv'
complete_file_name = os.path.join(DIRECTORY, file_name)
########
avoid_char_list = [',', '.', ';', '<', '>', '?', '-']
word_dict_path = 'word_dict_json.json'
seq_dict_path = 'seq_dict_json.json'
text_column = 'text'

def cleaning():
    print("main")
    aggregator = ''
    df = pd.read_csv(complete_file_name)
    file_to_read = list(df[text_column])
    with open(os.path.join(DIRECTORY, 'cleaned_for_the_net_{}'.format(file_name)), 'w') as file_to_write:
        for test_string in file_to_read:
            if test_string != "":

                # at least one pattern -> avoid
                result = False
                for pattern in pattern_list:
                    result = result or re.match(pattern, test_string)

                # title pattern example 'Inferno: CANTO XII'
                # title pattern or empty string
                if result or len(test_string) <= 3:
                    pass
                else:
                    # print(" curr string: {} ".format(test_string))
                    # avoid punctuations
                    for avoid_char in avoid_char_list:
                        test_string = test_string.replace(avoid_char, '')
                    # lower case
                    test_string = test_string.lower()
                    aggregator += test_string
                    file_to_write.write(test_string)
    return aggregator


def text_to_seq(aggregator):
    word_dict = {}
    seq_dict = {}
    counter = 0
    for word in aggregator.split():
        # add new words
        if word not in word_dict:
            word_dict.update({word: counter})
            seq_dict.update({counter: word})
            counter += 1

    return word_dict, seq_dict


# obtain a dict from json file
# read_dict_ = read_dict('word_dict_json.json')
#
def read_dict(path_to_json):
    with open(path_to_json, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    # to obtain the integer key as integer and not as string when needed
    parsed_json_int = json.loads(json_object,
                                 object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in
                                                        d.items()})
    return parsed_json_int


def features_labels(sequences, training_length=50):
    features = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in sequences:
        # print("seq {}".format(seq))
        # Create multiple training examples from each sequence
        for i in range(training_length, len(sequences)):
            # Extract the features and label
            extract = sequences[i - training_length:i + 1]
            # print("extract {}".format(extract))
            # Set the features and label
            features.append(extract[:-1])
            # the word after the sequence what we should predict
            labels.append(extract[-1])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def integers_conversion(aggregator):
    conversion_dict = read_dict(word_dict_path)
    converted_aggregator = []
    for word in aggregator.split():
        converted_aggregator.append(conversion_dict[word])
    return converted_aggregator


def create_one_hot_encoding(features, labels):
    # for each of our feature we create a row in the array
    # that contains 1 to the position of the label value
    # the array size is (n_examples,n_labels)
    # 0 0 1 this sample is labeled third
    # 0 1 0 this sample is labeled second

    n_words = len(features) + 1
    label_array = np.zeros((len(features), n_words), dtype=np.int8)
    for i, j in enumerate(labels):
        label_array[i, j] = 1
    return label_array


def main():
    aggregator = cleaning()
    print("cleaned")

    # first time need to be true, to create the dict
    need_to_create_word_dict = False
    if need_to_create_word_dict:

        word_dict, seq_dict = text_to_seq(aggregator)

        # save dict
        word_dict_json = json.dumps(word_dict)

        with open(word_dict_path, 'w') as f:
            json.dump(word_dict_json, f)

        # save dict
        seq_dict_json = json.dumps(seq_dict)
        with open(seq_dict_path, 'w') as f:
            json.dump(seq_dict_json, f)

    training_length = 10
    # features = features_labels_with_strings(aggregator, training_length)
    fixed_length_shorter_makes_sense = 500
    aggregator = aggregator[0:fixed_length_shorter_makes_sense]
    aggregator_integers = integers_conversion(aggregator)
    print("aggregator integers")
    # print(aggregator_integers)
    features, labels = features_labels(aggregator_integers,
                                       training_length)  # check if the features should be like these
    print(features, labels)
    label_array = create_one_hot_encoding(features, labels)
    print("a")
    # to obtain the label of the i = 3 index
    # i = 3
    # label_id = np.argmax(label_array[i])
    # seq_dict = read_dict(seq_dict_path)
    # seq_dict[label_id]




if __name__ == "__main__":
    main()

"""
def features_labels_with_strings(sequences, training_length=50):
    features = []
    labels = []

    sequences = sequences.split()
    # Iterate through the sequences of tokens
    for seq in sequences:
        # print("seq {}".format(seq))
        # Create multiple training examples from each sequence
        # for i in range(training_length, len(seq)):
        for i in range(training_length, len(sequences)):
            # Extract the features and label
            # extract = seq[i - training_length:i + 1]
            extract = sequences[i - training_length:i + 1]
            print("extract {}".format(extract))
            # Set the features and label
            features.append(extract[:-1])
            # the word after the sequence what we should predict
            labels.append(extract[-1])

    features = np.array(features)
    return features
"""
