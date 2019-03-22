# Active Learning for IC design by Ashish James, July 20, 2018
import os
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed


from sklearn.ensemble import RandomForestClassifier
# from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner, Committee

from models.model_clf import IC_Design_DNN_Clf

scaler = StandardScaler()
num_classes = 2
REG_FLAG = False
csv_file = "data/opAmp_280nm_GF55_Mod_30P.csv"  # Dataset1
input_dims = 9

def read_data(csv_file):
    data = pd.read_csv(csv_file)
    input = data.iloc[:,0:9]    
    if REG_FLAG:
        output = data.iloc[:,10:17]        
    else:
        output = data.iloc[:,9]
        integer_encoded = LabelEncoder().fit_transform(output)
        output = to_categorical(integer_encoded)
    return input, output
input, output = read_data(csv_file)

logdir = 'tf_logs/Inverse_Prob'
ckptdir = logdir + '/model'
if not os.path.exists(logdir):
    os.mkdir(logdir)

tf.reset_default_graph()
tf.random.set_random_seed(1)

with tf.name_scope('Classifier'):
    # Initialize neural network
    DNN = IC_Design_DNN_Clf('DNN')
    # Setup training process
    lmda = tf.placeholder_with_default(0.01, shape=[], name='lambda')
    X = tf.placeholder(tf.float32, [None, 9], name='X')
    Y = tf.placeholder(tf.float32, [None, 2], name='Y')

    tf.add_to_collection('placeholders', lmda)

    Targets = DNN(X)
    Targets_s = tf.nn.sigmoid(Targets)

    # cost = tf.reduce_mean(tf.square(Targets-Y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Targets, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    # correct_prediction = Targets - Y
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Targets_s, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    err_rate = 1 - accuracy

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()
input_norm = scaler.fit_transform(input)
scl_mean_ip = scaler.mean_
scl_var_ip = scaler.var_
# trainY = output #scaler.fit_transform(output)
# scl_mean_op = scaler.mean_
# scl_var_op = scaler.var_
# Train, test and validation datasets
trainX_tmp, testX, trainY_tmp, testY  = train_test_split(input_norm, output, test_size=0.2, random_state=1)
trainX_tmp, valX, trainY_tmp, valY = train_test_split(trainX_tmp, trainY_tmp, test_size=0.25, random_state=1)

sel_rate = 0.1

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


sel_met = 'random'
       # -----------------------------------
if sel_met == 'random':
    # --------------------------------------------------------------
    # No.1 Randomly select the train samples
    trainY_tmp_lab = np.argmax(trainY_tmp, axis=1)
    fail_ind = np.argwhere(trainY_tmp_lab == 0)[:,0]
    pass_ind = np.argwhere(trainY_tmp_lab == 1)[:,0]

    trainX_fail_tmp = trainX_tmp[fail_ind]
    trainY_fail_tmp = trainY_tmp[fail_ind]
    trainX_pass_tmp = trainX_tmp[pass_ind]
    trainY_pass_tmp = trainY_tmp[pass_ind]

    fail_ran = np.random.permutation(fail_ind.shape[0])
    fail_sel = fail_ran[np.arange(0, np.int(np.dot(fail_ind.shape[0], sel_rate)), 1)]
    pass_ran = np.random.permutation(pass_ind.shape[0])
    pass_sel = pass_ran[np.arange(0, np.int(np.dot(pass_ind.shape[0], sel_rate)), 1)]

    trainX_pass = trainX_pass_tmp[pass_sel]
    trainY_pass = trainY_pass_tmp[pass_sel]
    trainX_fail = trainX_fail_tmp[fail_sel]
    trainY_fail = trainY_fail_tmp[fail_sel]

    trainX = np.vstack((trainX_pass, trainX_fail))
    trainY = np.vstack((trainY_pass, trainY_fail))

else :
    #---------------------------------------------------
    #active learning  
    import pdb; pdb.set_trace()
    n_members = 2
    learner_list = list()

    for member_idx in range(n_members):
        # initial training data
        n_initial = 2
        initial_idx = np.random.choice(range(trainX_tmp.shape[0]), size=n_initial, replace=False)
        trainX_initial = trainX_tmp[initial_idx]
        trainY_initial = trainY_tmp[initial_idx][:,0]

        trainX_pool = np.delete(trainX_tmp, initial_idx, axis = 0)
        trainY_pool = np.delete(trainY_tmp[:,0], initial_idx)
        trainY_pool_org = np.delete(trainY_tmp, initial_idx, axis = 0)

        # initializing the active learner
        learner = ActiveLearner(
            estimator = RandomForestClassifier(),
            # query_strategy = uncertainty_sampling,   
            X_training = trainX_initial, y_training = trainY_initial
            )
        
        learner_list.append(learner)

    committee = Committee(learner_list = learner_list)

    # unqueried_score = committee.score(trainX_tmp,trainY_tmp[:,0])
    # performance_history = [unqueried_score]
    # active learning
    n_queries = np.int(np.dot(trainX_tmp.shape[0], sel_rate)) - n_initial

    trainX = np.zeros(shape = (n_queries, 9))
    trainY = np.zeros(shape = (n_queries, 2))
    for idx in range(n_queries):
        query_idx, query_instance = committee.query(trainX_pool)  
        print(query_idx)

        committee.teach(trainX_pool[query_idx].reshape(1,-1), trainY_pool[query_idx]) 
        # performance_history.append(committee.score(trainX_tmp, trainY_tmp[:,0]))
        trainX[idx] = trainX_pool[query_idx]
        trainY[idx] = trainY_pool_org[query_idx]
        trainX_pool, trainY_pool, trainY_pool_org = np.delete(trainX_pool, query_idx, axis = 0), np.delete(trainY_pool, query_idx, axis = 0), np.delete(trainY_pool_org, query_idx, axis = 0)
# --------------------------------------------------------------------------------
# Hyper parameters

training_epochs = 50
batch_size = 3

for epoch in range(training_epochs):
    total_batch = int(trainX.shape[0] / batch_size)
    
    avg_cost = 0
    avg_acc = 0

    for i in range(total_batch):
        ran_from = i * batch_size
        ran_to = (i + 1) * batch_size
        batch_xs = trainX[ran_from:ran_to]
        batch_ys = trainY[ran_from:ran_to]
        batch_ys = np.reshape(batch_ys, [batch_size, 2])
        # batch_ys = batch_ys.values.reshape(batch_size, 1)
        _, cc, aa, summary_str, tt, yy = sess.run([optimizer, cost, accuracy, summary, Targets, Y], feed_dict={X: batch_xs, Y: batch_ys})
        
        avg_cost += cc / total_batch
        avg_acc += aa / total_batch

        file_writer.add_summary(summary_str, epoch * total_batch + i)

    err_rate_val = sess.run(err_rate, feed_dict={X: valX, Y: valY})
    print('Epoch:', '%04d' % (epoch + 1), 'Tsize =', '%d' % trainX.shape[0], 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc), 'err_r =', '{:.9f}'.format(err_rate_val),)

acc_test = sess.run(accuracy, feed_dict={X: testX, Y: testY})
print('Accuracy_Test =', '{:.9f}'.format(acc_test))
    # saver.save(sess, ckptdir)
sess.close()

