import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav
import random
import math
from keras.utils import np_utils
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from keras.models import load_model
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave

# pre processing steps


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """
    Short-time Fourier transform of audio signal.
    """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)

    samples = np.append(sig, np.zeros((frameSize//2), dtype=int))
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize, dtype=int))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                               samples.strides[0])).copy()
    frames = frames.astype('float64')
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically.
    """
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    """
    A function that converts a wav file into a spectrogram represented by a \
    matrix where rows represent frequency bins, columns represent time, and \
    the values of the matrix represent the decibel intensity. A matrix of \
    this form can be passed as input to the CNN after undergoing normalization.
    """
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping

    return ims


def get_cropped_samples(matrix, crop_width=125):
    """
    Get N random samples with width of crop_width from the numpy matrix
    representing the participant's audio spectrogram.
    """
    # crop full spectrogram into segments of width = crop_width
    clipped_mat = matrix[:, (matrix.shape[1] % crop_width):]
    n_splits = clipped_mat.shape[1] / crop_width
    samples = np.split(clipped_mat, n_splits, axis=1)
    return samples


def preprocess(data):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    data = np.array(data)
    print(type(data))
    data = data.astype('float32')
    data = np.array([(X - X.min()) / (X.max() - X.min()) for X in data])
    return data


def prep_train_test(data):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    # normalize to dBfS
    data = preprocess(data)

    # Convert class vectors to binary class matrices
    data = np_utils.to_categorical(data)

    return data


def keras_img_prep(data, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    data = data.reshape(2*data.shape[0], img_rows, img_cols, 1)
    return data


def create_sample_dictionary(audio_path):
    data = stft_matrix(audio_path)
    data_samples = get_cropped_samples(data)
    data_samples = prep_train_test(data_samples)
    print(np.shape(data_samples))
    img_rows, img_cols = data_samples.shape[1], data_samples.shape[2]
    data_samples = keras_img_prep(data_samples, img_rows, img_cols)
    return data_samples


def remove_silence(filename, smoothing=1.0, weight=0.3, plot=False):
    # create participant directory for segmented wav files
    [Fs, x] = aIO.read_audio_file(filename)
    segments = aS.silence_removal(x, Fs, 0.020, 0.020,
                                  smooth_window=smoothing,
                                  weight=weight,
                                  plot=plot)
    data = []
    for s in segments:
        seg_name = "segment_{:.2f}-{:.2f}.wav".format(s[0], s[1])
        wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])
    files = os.listdir(os.curdir)
    for file in files:
        if file.endswith('.wav'):
            w = wave.open(file, 'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
            os.remove(file)
    # print(filename)
    output = wave.open(filename, 'wb')
    output.setparams(data[0][0])
    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()

def get_face(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=4,
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        cv2.imwrite(r'./face/face.jpg', face)


app = Flask(__name__)
CORS(app)

model_dep = load_model(r'C:\Users\Hadi\Desktop\Emily_project\emily\Depression\model-dep.h5', compile=True)
model_fer = load_model(r'C:\Users\Hadi\Desktop\Emily_project\emily\mde\model-fer.h5', compile=True)

fer = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"netural",
    5:"sad",
    6:"surprise",
}

@app.route('/classify-dep', methods=["POST"])
def index_dep():
    print(request.files)
    recieved = request.files['file']
    recieved.save(r'./'+recieved.filename)
    remove_silence(r'./'+recieved.filename)
    data = create_sample_dictionary(
        r'./'+recieved.filename)
    prediction = model_dep.predict(data)
    prediction = np.argmax(prediction, axis=1)
    print(prediction)
    dep = 0
    net = 0
    return 

@app.route('/classify-fer', methods=["POST"])
def index_fer():
    print(request.files)
    recieved = request.files['file']
    recieved.save(r'./img/'+recieved.filename)
    get_face(os.path.abspath(r'./img/'+recieved.filename))
    img = cv2.imread(r'./face/face.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape((1, 48, 48, 1))
    print(np.shape(img))
    img = tf.keras.preprocessing.image.img_to_array(img)
    prediction = model_fer.predict(img)
    print(prediction)
    prediction = np.argmax(prediction, axis=1)
    return {
        "exp":str(fer[prediction[0]])
    }

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
