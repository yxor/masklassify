import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from util import plot_history
import numpy as np
import json


def load_data(filename : str):
    with open(filename, "r") as f:
        data = json.load(f)
    inputs = np.array(data["inputs"])
    labels = np.array(data["labels"])

    return inputs, labels


if __name__ == "__main__":

    train_inputs, train_labels = load_data("training_0.json")
    test_inputs, test_labels = load_data("validation_0.json")
    num_epochs = 1000

    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(train_inputs.shape[1], train_inputs.shape[2])),

        # 1st dense layer
        keras.layers.Dense(126, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        keras.layers.Dropout(0.5),

        # 1st dense layer
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),

        # output layer
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compiling the model
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=['acc'])

    model.summary()

    # train model
    history = model.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), batch_size=16, epochs=num_epochs)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    # visualizing learning evolution
    plot_history(history)
