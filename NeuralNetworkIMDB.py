import sys

import tensorflow as td
import value as value
from keras.datasets import imdb
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

model = keras.Sequential()


(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def ask_to_open_file():
    open_from_file = input(str("load from file?  T/F  ")).capitalize()
    if open_from_file == "T":
        try:
            global model
            model = (keras.models.load_model("model.h5"))
            ask_for_number()
        except FileNotFoundError as e:
            print("No file available. Will now retrain the model")
            train_model()
    elif open_from_file == "F":
        train_model()
    else:
        print("Invalid input. Please enter only T or F")
        ask_to_open_file()


def train_model():
    global model
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(25, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fit_model = model.fit(x_train, y_train, epochs=200, batch_size=1024, validation_data=(x_val, y_val), verbose=2)
    results = model.evaluate(test_data, test_labels)
    print(results)
    model.save("model.h5")
    ask_for_number()


def ask_for_number():
    num = input(str("Choose a number or F to exit: "))

    try:
        if num.capitalize() == "F":
            print("Exiting now")
            sys.exit(1)
        else:

            test_review = test_data[int(num)]
            predict = model.predict(test_data[int(num)])
            print("Review: ")
            print(decode_review(test_data[int(num)]))
            print("Prediction: " + str(predict[int(num)]))
            print("Actual: " + str(test_labels[int(num)]))
            ask_for_number()

    except Exception as e:
        print("Invalid input")
        ask_for_number()





ask_to_open_file()