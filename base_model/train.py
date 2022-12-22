# This file trains the GNN. It roughly follows the format proposed in:
# https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/qm9_ecc.py

from base_model import base_model

import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
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
    default=1e-4,
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
    default=16,
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
from SpektralDataset import train_data
from SpektralDataset import valid_data

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
        epochs+=1
        loss = 0

################
# Evaluate model
################
print("Testing model")
loss = 0

def T_constructor(u_s, u_v):
    Sigma = tf.math.abs(tf.linalg.diag(u_s))
    return tf.linalg.matmul(tf.linalg.matmul(u_v, Sigma), tf.transpose(u_v, perm=[0, 2, 1]))

for i, batch in enumerate(loader_vl):
    inputs, target = batch

    y_s = target[:, :, 3]
    y_v = target[:, :, 0:3]
    y = T_constructor(y_s, y_v)

    predictions = model(inputs, training=False)
    u_s = predictions[:, :, 3]
    u_v = predictions[:, :, 0:3]
    pred = T_constructor(u_s, u_v) # no more exp

    loss += loss_fn(y, pred)
    
loss /= loader_vl.steps_per_epoch
print("Done. Test loss: {}".format(loss))
print("################\n")
