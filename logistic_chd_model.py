#Wilson Burchenal
#Stanford tf tutorial Assignment 1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics import roc_auc_score

learning_rate = 0.0001
num_epochs = 100

#Read data in from csv files
df = pd.read_csv('./data/heart.csv')
msk = np.random.rand(len(df)) <= 0.7
train = df[msk]
test = df[~msk]

train_size = train.shape[0]
test_size = test.shape[0]

print(train_size)
print(test_size)

# print(train.drop('chd', 1))

#Placeholders
x_tr = tf.placeholder(dtype=tf.float32, shape=[train_size, 8], name="inputs")
y_tr = tf.placeholder(dtype=tf.float32, shape=[train_size, 1], name="labels")

x_te = tf.placeholder(dtype=tf.float32, shape=[test_size, 8], name="inputs")
y_te = tf.placeholder(dtype=tf.float32, shape=[test_size, 1], name="labels")

# Weights and Bias
w = tf.Variable(tf.random_normal(shape=[8,1], stddev=0.01), name="weights")
# b = tf.Variable(tf.zeros(shape=[1,10]), name="bias")

# Model
logits_tr = tf.matmul(x_tr,w)
logits_te = tf.matmul(x_te,w)

#Loss
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_tr, labels=y_tr, name="loss"))

#Gradient
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:

	# Train the model
	start_time=time.time()
	sess.run(tf.global_variables_initializer())

	for i in range(num_epochs):
		X_train = train.drop('chd',1).drop('famhist',1)
		Y_train = train['chd'].values.reshape(train_size, 1)

		# print(X_train.shape)
		# print(Y_train.shape)

		_, loss_train = sess.run([optimizer, loss], feed_dict={x_tr: X_train, y_tr: Y_train})

		print('Loss for epoch {0}: {1}'.format(i, loss_train))

	print('Total time: {0} seconds'.format(time.time() - start_time))


	# Test the model
	preds = tf.nn.sigmoid(logits_tr)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y_te, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	total_correct_preds = 0
	X_test = test.drop('chd',1).drop('famhist',1)
	Y_test = test['chd'].values.reshape(test_size, 1)

	accuracy_batch = sess.run([accuracy], feed_dict={x_te: X_test, y_te:Y_test}) 
	total_correct_preds = sum(accuracy_batch)
	
	print('Accuracy {0}'.format(total_correct_preds/test_size))


print("Praise the sun")