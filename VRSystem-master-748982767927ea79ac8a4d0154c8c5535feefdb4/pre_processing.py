import os
from os import listdir
from os.path import isfile, join
import time
import numpy as np
import dill
import librosa
from python_speech_features import mfcc

SPACE_TOKEN = '<space>'
FIRST_INDEX = ord('a') - 1
SPACE_INDEX = 0

datapath = os.path.dirname(os.path.abspath(__file__))

captionpath = os.path.join(datapath, 'captions/')
audiopath = os.path.join(datapath, 'audio/')
cachepath = os.path.join(datapath, 'cache/')

videoIds = [f.replace('.txt','') for f in listdir(captionpath) if isfile(join(captionpath, f))]

def next_video(videoIndex):
    video_id=videoIds[videoIndex]
    captionfile=captionpath+video_id+'.txt'
    audiofile=audiopath+video_id+'.wav'
    return video_id, captionfile, audiofile

def audio_preporcessor(audiofile, starttime, duration):
    audio, _ = librosa.load(audiofile, sr=8000, mono=True, offset=starttime, duration=duration)
    audio = audio.reshape(-1, 1)
    inputs = mfcc(audio, samplerate=8000)

    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    train_seq_len = [train_inputs.shape[1]]

    return train_inputs, train_seq_len

def sparse_tuple_from(sequences, dtype=np.int32):
    #######################################################################################
    # Authored by Igor Macedo Quintanilha, Retrieved from https://github.com/igormq/ctc_tensorflow_example/blob/master
    # /utils.py , September 2017)
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    #######################################################################################
    return indices, values, shape

def caption_preprocessor(target_text):
    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '')\
        .replace("'", '').replace('!', '').replace('-', '').replace('[Music]', '')

    targets = original.replace(' ', '  ')
    targets = targets.split(' ')
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

    train_targets = sparse_tuple_from([targets])
    return train_targets

if __name__ == '__main__':
    for videoIndex in range(len(videoIds)):
        start = time.time()
        video_id, captionfile, audiofile=next_video(videoIndex)
        caption = open(captionfile, 'r',encoding='UTF-8').read().strip()
        caption=caption.split('\n')
        array=[]
        for i in range(len(caption)):
            try:
                captionline=caption[i]
                index, starttime, duration, target_text = captionline.split('--')
                train_targets  = caption_preprocessor(target_text)
                train_inputs, train_seq_len = audio_preporcessor(audiofile, float(starttime), float(duration))
                obj = {
                    'train_inputs': train_inputs,
                    'train_targets': train_targets,
                    'train_seq_len': train_seq_len,
                    'original': target_text
                }
                array.append(obj)
            except Exception as e:
                print(e)
        cache_filename=cachepath+video_id+'_cache.pkl'
        with open(cache_filename, 'wb') as f:
            dill.dump(array, f)
        end = time.time()
        duration=end-start
        print('Cache for '+video_id+' created')
        print('duration',duration)

