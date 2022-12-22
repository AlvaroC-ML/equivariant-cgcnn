from SpektralDataset_gvp import valid_data
from spektral.data import DisjointLoader
from model import anisotropy

import tensorflow as tf

import numpy as np
import random as rand

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

# u_s.shape = [batch, 3] 
 # u_v.shape = [batch, 3, 3]
def T_constructor(u_s, u_v):
    Sigma = tf.linalg.diag(tf.math.abs(u_s))
    return tf.linalg.matmul(tf.linalg.matmul(u_v, Sigma), tf.transpose(u_v, perm=[0, 2, 1]))

batch_size = 4
epochs = 1

model = anisotropy()
loader=DisjointLoader(valid_data, batch_size=batch_size, epochs=epochs)

for batch in loader:
    g, y = batch
    x, a, e, i = g

    hi = model([x, a, e, i])

    # R(M(C))
    u_s1 = hi[:, :, 3]
    u_v1 = hi[:, :, :3]
    u_v1 = tf.linalg.matmul(R, u_v1)

    # Rotate crystal
    e_s = e[:, 3]
    e_v = e[:, 0:3]
    e_v = tf.linalg.matmul(e_v, tf.transpose(R))
    e = tf.concat([e_v, tf.expand_dims(e_s, axis = 1)], axis = 1)

    #M(R(C))
    hi = model([x, a, e, i])
    u_s2 = hi[:, :, 3]
    u_v2 = hi[:, :, :3]
    break

for inp in range(batch_size):
    print("batch")
    print(u_v1[inp])
    print(u_v2[inp])
