#/usr/bin/python3

import numpy as np

import random
from time import sleep

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys

model = keras.models.load_model("./recog-model")

(x_train,y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

for k in range(0,10):
    print('-----------')
    print("Random sample image:")

    #Show off a random number
    sample = random.randint(0,5000)

    for i in range(len(x_test[sample])):
        for j in range(len(x_test[sample][i])):
            if x_test[sample][i][j] == 0: sys.stdout.write(' ')
            elif x_test[sample][i][j] < 100: sys.stdout.write('-')
            elif x_test[sample][i][j] < 255: sys.stdout.write('=')
            elif x_test[sample][i][j] == 255: sys.stdout.write('#')
        print()
    print("-------")

    #Shape the input
    x_test_feed = x_test.reshape(10000,784).astype("float32") / 255

    predictions = model.predict_on_batch(x_test_feed)

    #Print the prediction

    predict_pos = 0
    largest = -100

    for i in range(len(predictions[sample])):
        if predictions[sample][i] > largest:
            predict_pos = i
            largest = predictions[sample][i]
            #print("DEBUG:\t predict_pos: ", predict_pos, "\t predictoins[0][i]:", predictions[0][i])

    #print("DEBUG:\n",predictions[0])
    print("This image represents a...")
    print(predict_pos)
    print("-------------------------------------")
    sleep(2)
