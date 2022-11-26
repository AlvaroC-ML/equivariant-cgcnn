from model_gvp import anisotropy

import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader

################
# Hyperparameters
################
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help="Learning rate"
)
parser.add_argument(
    '--epochs',
    type=int,
    default=25,
    help="Number of epochs"
)
parser.add_argument(
    '--batch',
    type=int,
    default=64,
    help="Batch size"
)
args=parser.parse_args()

learning_rate=args.lr
epochs=args.epochs
batch_size=args.batch

print("Learning rate", learning_rate)
print("Number of epochs", epochs)
print("Batch size", batch_size)

################
# Loading data
################
from SpektralDataset_gvp import train_data
from SpektralDataset_gvp import valid_data

loader_tr=DisjointLoader(train_data, batch_size=batch_size, epochs=epochs)
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

import time

tic = time.perf_counter()

step = loss = 0
for batch in loader_tr:
    step += 1
    loss += train_step(*batch)
    if step == loader_tr.steps_per_epoch:
        step = 0
        #print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
        loss = 0

toc = time.perf_counter()

print(f"Time to train:{toc-tic}")

################
# Evaluate model
################
print("Testing model")
loss = 0
for batch in loader_vl:
    inputs, target = batch
    predictions = model(inputs, training=False)
    loss += loss_fn(tf.math.exp(target), tf.math.exp(predictions))
loss /= loader_vl.steps_per_epoch
print("Done. Test loss: {}".format(loss))
print("################\n")