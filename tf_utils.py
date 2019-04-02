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
import time
from numpy.random import seed
from vogn_utils import train_model_cc,train_model_bb,csvDataset,assist,ToTensor
seed(1)
from tensorflow import set_random_seed

import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
from models import SimpleConvNet
from datasets import Dataset
import torch.nn.functional as F
sns.set()
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
# from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner, Committee

##from models.model_clf import IC_Design_DNN_Clf

scaler = StandardScaler()
num_classes = 2
REG_FLAG = False
csv_file = "circuit-design/opAmp_280nm_GF55_Mod_30P.csv"  # Dataset1
input_dims = 9

import tensorflow as tf


class IC_Design_DNN_Clf:

    def __init__(self, name):
        self.name = name

    def __call__(self, X, reuse=False):

        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            dense1 = tf.layers.dense(inputs=X, units=64, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.relu)
            # dense3 = tf.layers.dense(inputs=dense2, units=64, activation=tf.nn.relu)
            outputs = tf.layers.dense(inputs=dense2, units=2, activation=tf.nn.relu)

        return outputs

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

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

#logdir = 'tf_logs/Inverse_Prob'
#ckptdir = logdir + '/model'
#if not os.path.exists(logdir):
#    os.mkdir(logdir)
def test_perf(samples):
    
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

    trainX=trainX_tmp[samples]
    trainY=trainY_tmp[samples]

    start = time.time()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
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
            # batch_ys = np.reshape(batch_ys, [batch_size, 2])
            # batch_ys = batch_ys.values.reshape(batch_size, 1)
            _, cc, aa, summary_str, tt, yy = sess.run([optimizer, cost, accuracy, summary, Targets, Y], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cc / total_batch
            avg_acc += aa / total_batch

            #file_writer.add_summary(summary_str, epoch * total_batch + i)

        err_rate_val = sess.run(err_rate, feed_dict={X: valX, Y: valY})
        if epoch%20==0:
            print('Epoch:', '%04d' % (epoch + 1), 'Tsize =', '%d' % trainX.shape[0], 'cost =', '{:.9f}'.format(avg_cost), 'acc =', '{:.9f}'.format(avg_acc), 'err_r =', '{:.9f}'.format(err_rate_val),)

        acc_test = sess.run(accuracy, feed_dict={X: testX, Y: testY})
        print('Accuracy_Test =', '{:.9f}'.format(acc_test))
        # saver.save(sess, ckptdir)

    sess.close()

    end = time.time()
    print (end - start)
    
    
def test_perf_py(samples,model,optimizer,X,Y):
    
    use_cuda = torch.cuda.is_available()
    inference_dataset = csvDataset(X[5529:],Y[5529:],transform= ToTensor())
    inference_loader = torch.utils.data.DataLoader(inference_dataset,batch_size=1383, shuffle=False)
    file_dataset = csvDataset(X[samples],Y[samples],transform= ToTensor())
    final_loader = torch.utils.data.DataLoader(file_dataset,batch_size=len(samples), shuffle=False)
    
    if use_cuda:
        model = model.float().cuda()
        
    criterion = F.binary_cross_entropy_with_logits
    model, train_loss, train_accuracy = train_model_cc(model, [final_loader, final_loader], criterion,
    optimizer, num_epochs=50)
    
    model.eval()
    with torch.no_grad():
        for i in inference_loader:
            inputs = i['data']
            labels = i['label']
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            out = model.forward(inputs)
            pred = (out.cpu().numpy()>0)*1.
            labels = (labels.cpu().numpy())*1.
        
    correct =(np.sum(pred==labels)/1383)
    
    print(correct)
