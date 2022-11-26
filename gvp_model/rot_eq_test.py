from SpektralDataset_gvp import valid_data
from spektral.data import DisjointLoader
from model_gvp import anisotropy

import tensorflow as tf

import numpy as np
import random as rand

##################
# Creating rotation matrix
##################

pi = np.pi

den1=rand.random() * 5
den2=rand.random() * 5
den3=rand.random() * 5

theta1 = pi/den1
theta2 = pi/den2
theta3 = pi/den3

R1 = tf.constant(
    [[1.0, 0.0, 0.0],
    [0.0, np.cos(theta1), np.sin(theta1)],
     [0.0, -np.sin(theta1), np.cos(theta1)]
    ], dtype = tf.float32)

R2 = tf.constant(
    [[np.cos(theta2), 0.0, np.sin(theta2)],
    [0.0, 1, 0.0],
     [-np.sin(theta2), 0.0, np.cos(theta2)]
    ], dtype = tf.float32)

R3 = tf.constant(
    [[np.cos(theta3), np.sin(theta3), 0.0],
     [-np.sin(theta3), np.cos(theta3), 0.0],
     [0.0, 0.0, 1.0]
    ], dtype = tf.float32)

R = np.matmul(np.matmul(R1, R2), R3)

############
#
############

batch_size = 4
epochs = 1

model = anisotropy()
loader=DisjointLoader(valid_data, batch_size=batch_size, epochs=epochs)

for batch in loader: # Get a batch out
    g, y = batch
    x, a, e, i = g

    test = model([x, a, e, i])

    # R(M(C))
    u_s1 = test[:, :, 0]
    u_v1 = test[:, :, 1:]
    u_v1 = tf.linalg.matmul(R, u_v1)

    # Rotate crystal
    e_s = e[:, 3]
    e_v = e[:, 0:3]
    e_v = tf.linalg.matmul(e_v, tf.transpose(R))
    e = tf.concat([e_v, tf.expand_dims(e_s, axis = 1)], axis = 1)

    # M(R(C))
    test = model([x, a, e, i])
    u_s2 = test[:, :, 0]
    u_v2 = test[:, :, 1:]
    break

# Print to check they are equal
for inp in range(batch_size):
    print("Batch:")
    print(u_v1[inp])
    print(u_v2[inp])