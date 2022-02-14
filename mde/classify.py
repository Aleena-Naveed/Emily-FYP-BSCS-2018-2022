import cv2
import os

from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import tensorflow as tf
app = Flask(__name__)
CORS(app)

model_fer = load_model(r'C:\Users\Hadi\Desktop\Emily_project\emily\mde\model-fer.h5', compile=True)


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


@app.route('/classify-fer', methods=["POST"])
def index():
    print(request.files)
    recieved = request.files['file']
    recieved.save(r'./img/'+recieved.filename)
    get_face(os.path.abspath(r'./img/'+recieved.filename))
    # data = ImageDataGenerator()
    # img = data.flow_from_directory(r"./face",
    #                                target_size=(48, 48),
    #                                color_mode="grayscale",
    #                                batch_size=1,
    #                                class_mode='categorical')
    img = cv2.imread(r'./face/face.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.reshape((1, 48, 48, 1))
    print(np.shape(img))
    img = tf.keras.preprocessing.image.img_to_array(img)
    prediction = model_fer.predict(img)
    print(prediction)
    prediction = np.argmax(prediction, axis=1)
    return


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
