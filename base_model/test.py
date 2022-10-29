from base_model import base_model

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader

################
# Hyperparameters
################
learning_rate = 1e-3  # Learning rate
epochs = 25  # Number of training epochs
batch_size = 32  # Batch size

################
# Loading data
################
from SpektralDataset import train_data
from SpektralDataset import valid_data

loader_tr=DisjointLoader(valid_data, batch_size=batch_size, epochs=epochs)
loader_vl=DisjointLoader(valid_data, batch_size=batch_size, epochs=1)

################
# Building model
################
model=base_model()
optimizer=Adam(learning_rate)
loss_fn=MeanSquaredError()

################
# Fit model
################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

step = loss = 0
for batch in loader_tr:
    step += 1
    loss += train_step(*batch)
    if step == loader_tr.steps_per_epoch:
        step = 0
        print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
        loss = 0

################
# Evaluate model
################
print("Testing model")
loss = 0
for batch in loader_vl:
    inputs, target = batch
    predictions = model(inputs, training=False)
    loss += loss_fn(target, predictions)
loss /= loader_vl.steps_per_epoch
print("Done. Test loss: {}".format(loss))