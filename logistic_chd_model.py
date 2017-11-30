#Wilson Burchenal
#Stanford tf tutorial Assignment 1

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics import roc_auc_score

learning_rate = 0.01
num_epochs = 100
batch_size = 10

#Read data in from csv files
df = pd.read_csv('./data/heart.csv')
msk = np.random.rand(len(df)) <= 0.7
train = df[msk]
test = df[~msk]

train_size = train.shape[0]
test_size = test.shape[0]

print(train_size)
print(test_size)


# test_dict = train.drop('chd',1)[:batch_size]
# print(test_dict)

#Placeholders
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 8], name="inputs")
y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1], name="labels")

# Weights and Bias
w = tf.Variable(tf.random_normal(shape=[8,1], stddev=0.01), name="weights")
# b = tf.Variable(tf.zeros(shape=[1,10]), name="bias")

# Model
logits = tf.matmul(x,w)

#Loss
loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y, name="loss"))

#Gradient
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:

	# Train the model
	start_time=time.time()
	sess.run(tf.global_variables_initializer())
	num_batches = int(train_size/batch_size)

	for i in range(num_epochs):
		total_loss = 0

		for _ in range(num_batches):

			X_batch = train.drop('chd',1).drop('famhist',1)[:batch_size]
			Y_batch = train['chd'].values.reshape(train_size, 1)[:batch_size]

			_, loss_train = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch})
			total_loss += loss_train

		print('Loss for epoch {0}: {1}'.format(i, loss_train))

	print('Total time: {0} seconds'.format(time.time() - start_time))


	# Test the model
	preds = tf.nn.sigmoid(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	total_correct_preds = 0
	
	num_batches = int(test_size/batch_size)

	for i in range(num_batches):
		X_test = train.drop('chd',1).drop('famhist',1)[:batch_size]
		Y_test = train['chd'].values.reshape(train_size, 1)[:batch_size]
		accuracy_batch = sess.run([accuracy], feed_dict={x: X_test, y:Y_test}) 
		total_correct_preds += sum(accuracy_batch)
	
	print('Accuracy {0}'.format(total_correct_preds/test_size))


print("Praise the sun")