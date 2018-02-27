from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
import pickle
import dill
import numpy as np
import time
import tensorflow as tf
from oauth2client.tools import argparser

num_features = 13
num_layers = 1
num_hidden = 100
num_classes = ord('z') - ord('a') + 1 + 1 + 1

num_epochs = 20
batch_size = 1

datapath = os.path.dirname(os.path.abspath(__file__))
cachepath = os.path.join(datapath, 'cache/')
modelpath = os.path.join(datapath, 'models/')

cachefiles = [f for f in listdir(cachepath) if isfile(join(cachepath, f))]

def next_cache(cacheIndex):
    cachefile=cachefiles[cacheIndex]
    with open(cachepath+cachefile, 'rb') as f:
        cache = pickle.load(f)
    return cache

def delete_cache_lines(cacheIndex,linesToBeDeleted):
    cachefile=cachefiles[cacheIndex]
    with open(cachepath+cachefile, 'rb') as f:
        cache = pickle.load(f)
    new_cache = np.delete(cache, linesToBeDeleted)
    with open(cachepath+cachefile, 'wb') as f:
        dill.dump(new_cache, f)

def trining():
    #######################################################################################
    # Authored by Igor Macedo Quintanilha,
    # Retrieved from https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py ,
    # September 2017
    graph = tf.Graph()
    with graph.as_default():
        audio_input = tf.placeholder(tf.float32, [None, None, num_features])
        text_input = tf.sparse_placeholder(tf.int32)
        seq_len = tf.placeholder(tf.int32, [None])

        lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        stack = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        outputs, laststate = tf.nn.dynamic_rnn(stack, audio_input, seq_len, dtype=tf.float32)
        shape = tf.shape(audio_input)
        batch_s, max_timesteps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        predictions = tf.matmul(outputs, W) + b
        predictions = tf.reshape(predictions, [batch_s, -1, num_classes])
        predictions = tf.transpose(predictions, (1, 0, 2))

        loss = tf.nn.ctc_loss(text_input, predictions, seq_len)
        cost = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(cost)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(predictions, seq_len)

        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),text_input))
    #######################################################################################
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        try:
            new_saver = tf.train.import_meta_graph(modelpath+'model.meta')
            new_saver.restore(session,tf.train.latest_checkpoint('models/'))

        except Exception as e:
            tf.global_variables_initializer().run()

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            for cacheIndex in range(len(cachefiles)):
                start = time.time()
                cache=next_cache(cacheIndex)
                print("training",cachefiles[cacheIndex])
                linesToBeDeleted=[]
                lineIndex=0
                while(lineIndex<len(cache)):
                    try:
                        train_inputs=cache[lineIndex]['train_inputs']
                        train_targets=cache[lineIndex]['train_targets']
                        train_seq_len=cache[lineIndex]['train_seq_len']

                        feed = {audio_input: train_inputs,
                                text_input: train_targets,
                                seq_len: train_seq_len}
                        batch_cost, _ = session.run([cost, optimizer], feed)
                        train_cost += batch_cost
                        train_ler += session.run(ler, feed_dict=feed)
                    except Exception as e:
                        linesToBeDeleted.append(lineIndex)
                    lineIndex+=1
                if(curr_epoch==0):
                    try:
                        delete_cache_lines(cacheIndex,linesToBeDeleted)
                    except Exception as e:
                        print(e)
                train_cost /= len(cache)
                train_ler /= len(cache)
                log = "Epoch {}/{}, cache = {}/{} train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, num_epochs, cacheIndex+1, len(cachefiles), train_cost, train_ler,
                                 time.time() - start))
        save_path = saver.save(session, modelpath+"model")

if __name__ == '__main__':
    argparser.add_argument("--learn_rate",
                           help="Required; learning for training, values between 0.0001 and 0.001",
                           default=0.001)
    args = argparser.parse_args()
    initial_learning_rate=args.learn_rate
    initial_learning_rate=float(initial_learning_rate)
    trining()

