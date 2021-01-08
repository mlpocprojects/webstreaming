from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from models.model import load_pre_trained_model
import os
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# Load the model
BASE_DIR = os.getcwd()
model_path = os.path.join(BASE_DIR, 'models/weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
trained_model = load_pre_trained_model(model_path=model_path)

# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to warm up
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def web_stream():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        resized_frame = cv2.resize(frame, (224, 224))
        # apply the model
        # convert image to array
        x = image.img_to_array(resized_frame)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # predict class for image
        top_pred = trained_model.predict(x)
        # print('Result:', decode_predictions(top_pred, top=1)[0])
        results = decode_predictions(top_pred, top=1)[0]
        label = results[0][1]
        # print(label)
        cv2.putText(frame, label, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            output_resize = imutils.resize(outputFrame, width=680, height=680)
            (flag, encodedImage) = cv2.imencode(".jpg", output_resize)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start a thread that will perform web stream
    t = threading.Thread(target=web_stream)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host='0.0.0.0', port='8000', debug=True,
            threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
