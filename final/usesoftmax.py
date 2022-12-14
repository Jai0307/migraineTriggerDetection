import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray, random
from requests import session
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from load_data import loadData
labels, features = loadData()
print("labels: ", labels.shape)
print("features: ", features.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)
useSMOTE = True
if(useSMOTE):
    os = SMOTE(random_state=0)
    os_data_X,os_data_y=os.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=os_data_X)
    y_train = pd.DataFrame(data=os_data_y)
    print("length of oversampled data is ",len(os_data_X))
    print(y_train[0].value_counts())

y_train = np.array(y_train)

num_features = features.shape[1]
print('num_features: ', num_features)
num_labels = 6
learning_rate = 0.01
batch_size = 32
training_epochs = 10000
display_step = 100

def to_onehot(y):
    data = np.zeros((num_labels))
    data[int(y)] = 1
    return data

X_train = np.reshape(X_train, (-1, num_features))
X_test = np.reshape(X_test, (-1, num_features))

y_train = np.array([to_onehot(y) for y in y_train])
# print("y_train.shape: ", y_train)
y_test = np.array([to_onehot(y) for y in y_test])
# print("y_test.shape: ", y_test)


tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, num_features])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels])
a = tf.compat.v1.placeholder(tf.float32, shape=[1])
b = tf.compat.v1.placeholder(tf.float32, shape=[1])

# Weight
W = tf.Variable(tf.zeros([num_features, num_labels]))

# Bias
b = tf.Variable(tf.zeros([num_labels]))

intercept = True

# Construct model
if intercept==False:
    pred = tf.nn.softmax(tf.matmul(x, W)) # Softmax
else:
    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

pred_null = tf.ones([1, y.shape[1]]) / y.shape[1]

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(pred), axis=1))
cost_null = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(pred_null), axis=1))

def costFn(X, y, W, b):
    X = tf.cast(X, dtype='float32')
    if intercept:
        prediction = tf.nn.softmax(tf.matmul(X, W) + b)
    else:
        prediction = tf.nn.softmax(tf.matmul(X, W))
    c = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(prediction), axis=1))
    # print(sess.run([c]))
    return c

# Gradient Descent
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.compat.v1.global_variables_initializer()

def accuracy(predictions, labels): 
    # print(predictions)
    correctly_predicted1 = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) 
    pred2 = predictions

    pred2[:, np.argmax(predictions, 1)] = -999999
    correctly_predicted2 = 0
    # correctly_predicted2 = np.sum(np.argmax(pred2, 1) == np.argmax(labels, 1)) 
    acc = (100.0 * (correctly_predicted1 + correctly_predicted2)) / predictions.shape[0] 
    return acc

with tf.compat.v1.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        avg_acc = 0.0
        total_batch = len(X_train)//batch_size
        
        for i in range(total_batch):
            batch_x = X_train[i:i+1*batch_size]
            batch_y = y_train[i:i+1*batch_size]
            
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c/total_batch
            
            pred_y = sess.run(pred, feed_dict={x: batch_x})
            acc = accuracy(pred_y, batch_y)
            avg_acc += acc/total_batch
            
            
        if (epoch+1) % display_step == 0:
            tc = sess.run(cost, feed_dict={x: X_test, y: y_test})
            pred_y = sess.run(pred, feed_dict={x: X_test})
            ta = accuracy(pred_y, y_test)
            
            print("Epoch: {:2.0f} - Cost: {:0.5f} - Acc: {:0.5f} - Test Cost: {:0.5f} - Test Acc: {:0.5f}".format(
                epoch+1, avg_cost, avg_acc, tc, ta))
    
    print("Optimization Finshed")

    print("calculating statistics")
    c_n = sess.run(cost_null, feed_dict={y: y_test})
    print('cost null:', c_n)
    m, n = W.shape
    p = np.zeros([m, n])
    W_array = W.eval(sess)
    b_array = b.eval(sess)
    print(avg_cost)
        
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1))
    # Calculate accuracy
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test Accuracy:", acc.eval({x: X_test, y: y_test}))
    for i in range(m):
        for j in range(n):
            w_i_j = np.zeros(W.shape)
            b_i_j = np.zeros(b_array.shape)
            if i==0:
                b_i_j[j] = b_array[j]
                b = tf.convert_to_tensor(b_i_j, dtype='float32')
                W = tf.convert_to_tensor(w_i_j, dtype='float32')
            else:
                w_i_j[i, j] = W_array[i, j]
                b = tf.convert_to_tensor(b_i_j, dtype='float32')
                W = tf.convert_to_tensor(w_i_j, dtype='float32')

            b = tf.convert_to_tensor(b_i_j, dtype='float32')
            W = tf.convert_to_tensor(w_i_j, dtype='float32')
            c_i = costFn(X_train, y_train, W, b)
            chi_v = 2*(c_n - c_i)*X_train.shape[0]
            chi_val = sess.run([chi_v])
            # print('chi_val:', chi_val)
            p[i, j] = stats.chi2.sf(chi_val, (m-1)*n)
print("R2: ", (c_n - avg_cost) / c_n )
print(p)
