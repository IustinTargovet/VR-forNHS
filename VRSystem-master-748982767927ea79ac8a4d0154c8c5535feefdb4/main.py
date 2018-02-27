import os
from flask import Flask, render_template, flash, request, url_for, redirect, session
import librosa
from python_speech_features import mfcc
import numpy as np
import tensorflow as tf

app = Flask(__name__)
datapath = os.path.dirname(os.path.abspath(__file__))
modelpath = os.path.join(datapath, 'models/')

num_features = 13
num_classes = ord('z') - ord('a') + 1 + 1 + 1
initial_learning_rate=1
num_hidden = 100
num_layers = 1
FIRST_INDEX = ord('a') - 1

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route("/upload", methods=['POST'])
def upload():
    try:
        audiofolder = os.path.join(datapath, 'tmp/')

        file = request.files['file']
        filename = file.filename

        destination = "/".join([audiofolder, filename])
        file.save(destination)

        audio, _ = librosa.load(destination, sr=8000, mono=True)
        audio = audio.reshape(-1, 1)
        inputs = mfcc(audio, samplerate=8000)

        train_inputs = np.asarray(inputs[np.newaxis, :])
        train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
        train_seq_len = [train_inputs.shape[1]]

        os.remove(destination)

        graph = tf.Graph()
        with graph.as_default():
            #######################################################################################
            # Authored by Igor Macedo Quintanilha,
            # Retrieved from https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py ,
            # September 2017
            audio_input = tf.placeholder(tf.float32, [None, None, num_features])
            text_input = tf.sparse_placeholder(tf.int32)
            seq_len = tf.placeholder(tf.int32, [None])

            lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            stack = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers,state_is_tuple=True)
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
            saver = tf.train.Saver()
            #######################################################################################

        with tf.Session(graph=graph) as session:
            # Initializate the weights and biases
            tf.global_variables_initializer().run()
            new_saver = tf.train.import_meta_graph(modelpath+'model.meta')
            new_saver.restore(session,tf.train.latest_checkpoint('models/'))

            feed = {audio_input: train_inputs,
                    seq_len: train_seq_len}

            d = session.run(decoded[0], feed_dict=feed)
            str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
            str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
            print('Decoded: %s' % str_decoded)
        return render_template("complete.html",string_decoded=str_decoded)
    except Exception as e:
        return str(e)

@app.route('/download_txt/<string:str_decoded>')
def download_txt(str_decoded):
    try:
        with open('text/transcript.txt', 'w') as txtfile:
            txtfile.write(str_decoded)
        return "download complete!"
    except Exception as e:
        return str(e)

if __name__ == '__main__':
  app.run()

