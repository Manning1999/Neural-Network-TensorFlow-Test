import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Main:

    def __init__(self):
        self.ask_to_open_file()

    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(28, 28)),

        # hidden layer
        keras.layers.Dense(128, activation="relu"),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    data = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    class_names = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    def ask_to_open_file(self):
        open_from_file = input(str("load from file?  T/F  ")).capitalize()
        if open_from_file == "T":
            try:
                loaded_model = pickle.load(open('saved model.sav', 'rb'))
                self.model.set_weights(loaded_model)
                self.ask_for_image_number()
            except:
                print("No file available. Will now retrain the model")
                self.retrain_model()
        elif open_from_file == "F":
            self.retrain_model()
        else:
            print("Invalid input. Please enter only T or F")
            self.ask_to_open_file()

    def retrain_model(self):
        print("Training model now")

        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])

        self.model.fit(self.train_images, self.train_labels, epochs=10)

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)

        print("Tested acc: ", test_acc)
        pickle.dump(self.model.weights, open('saved model.sav', 'wb'))

        self.ask_for_image_number()

    def ask_for_image_number(self):
        num = input(str("Enter a number: "))
        converted_num = int(num)
        print("will now guess what image ", num, "is")
        prediction = self.model.predict(self.test_images)
        plt.grid(False)
        plt.imshow(self.test_images[converted_num], cmap=plt.cm.binary)
        plt.xlabel("actual: " + self.class_names[self.test_labels[converted_num]])
        plt.title("Prediction: " + self.class_names[np.argmax(prediction[converted_num])])
        plt.show()
        self.ask_for_image_number()


Main()