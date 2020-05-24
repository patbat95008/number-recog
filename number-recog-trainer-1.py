#/usr/bin/python3

import numpy as np

#import pydot

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Building the AI Model...\n")

#
#Building the AI's mind
#
#Set up the Input layer, 784 inputs
inputs = keras.Input(shape=(784,))

#Create a "dense" 64 neuron layer
dense = layers.Dense(64, activation="relu")

#Link the Input layer to the first dense layer
# Inputs (784) -> NN Layer (64)
x = dense(inputs)

#Add an additional hidden layer
# Inputs (784) -> NN Layer 1 (64) -> NN Layer 2 (64)
x = layers.Dense(64, activation="relu")(x)

#Finally, create the output layer
# Inputs (784) -> NN Layer 1 (64) -> NN Layer 2 (64) -> Output (10)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

#Print the construction efforts
print("\n\nInputs:\t" + str(inputs.shape) + ", " + str(inputs.dtype))
print("\nOutput:\t" + str(outputs))

print(model.summary())
print("==========")

# ToDo: Get pydot to print an image of the AI's mind
#tmp = keras.utils.plot_model(model, "My_first_AI.png"), show_shapes=True

#
# Training
#

#Load the raw training data
#A 28 x 28 B&W image of a hand-written digit, each pixel can range from
#0 - 255 shades of grey
#The Training set is 60,000 images, the Test set is 10,000 images
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

#Shape the input data to be a 1-D array (28 x 28 = 784)
#Convert the 0-255 color value to a float value ranging from 0 - 1
x_train = x_train.reshape(60000,784).astype("float32") / 255
x_test = x_test.reshape(10000,784).astype("float32")/255

#Compile the model!
model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
)

#Begin the training 
history = model.fit(x_train, #Input data
        y_train, #Expected output data
        batch_size=64,
        epochs=40, #How many times to go over the data set
        validation_split=0.2, validation_data=(x_test,y_test),
        verbose=1 #Print the training progress
        )

test_scores = model.evaluate(x_test,y_test, verbose=2)

print("\n\nTest loss:\t", test_scores[0])
print("Test accuracy:\t",test_scores[1])

print("\n\n======\nSaving the model...")

#
#Save the trained model
#
#Saves the state of the model after training.
#Training code not needed, contains model arch, weights, compile args

model.save("./recog-model")

print("Model saved :^D\n\n===")
print("To load this model:") 
print(' model = keras.models.load_model("./numb-recog-model")')
