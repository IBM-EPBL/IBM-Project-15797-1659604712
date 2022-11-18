import os
import shutil
import threading
import time
from keras.models import model_from_json
import cv2
import operator
import numpy as np
from flask import Flask, flash, redirect, render_template, Response, request
app = Flask(__name__)

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

canStart = False
uploadsPath = "uploads"
totalImages = 0
outputImages = []
outputImageEdited = ''
predicted = ""


app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
if os.path.isdir(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSION = 'jpg'


def getRoiCoordinates(frame):
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 1)
    return [x1, x2, y1, y2]


def getRoi(frame):
    x1, x2, y1, y2 = getRoiCoordinates(frame)
    roi = frame[y1:y2, x1:x2]
    return roi


def filterImage(img):
    # skin tone lower and upper bound values in hsv
    lowerHSV = np.array([0, 30, 20])
    upperHSV = np.array([50, 255, 255])
    img = cv2.resize(img, (64, 64))
    kernel = np.ones((2, 2), np.uint8)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filtered = cv2.inRange(imgHsv, lowerHSV, upperHSV)
    filtered = cv2.erode(filtered, kernel, iterations=1)
    filtered = cv2.bitwise_not(filtered)
    # below two lines are alternative to the above 4 lines of code
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, filtered = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return filtered


def cameraRenderer():
    while True:
        if not canStart:
            continue
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        getRoiCoordinates(frame)
        cv2.putText(frame, predicted, (10, 120),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def outputRenderer():
    global outputImageEdited
    canBrowse = False
    i = 0
    while True:
        if not canStart:
            continue
        if (canBrowse):
            if (predicted == "1"):
                outputImageEdited = cv2.rotate(
                    outputImageEdited, cv2.ROTATE_90_CLOCKWISE)
            if (predicted == "2"):
                outputImageEdited = cv2.rotate(
                    outputImageEdited, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if (predicted == "3"):
                i += 1
                if (i >= totalImages):
                    i = 0
                outputImageEdited = outputImages[i]
            if (predicted == "4"):
                i -= 1
                if (i < 0):
                    i = totalImages-1
                outputImageEdited = outputImages[i]
            if (predicted == "5"):
                outputImageEdited = outputImages[i].copy()
            canBrowse = False
        if (predicted == "0"):
            canBrowse = True
        ret, buffer = cv2.imencode('.jpg', outputImageEdited)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def filterRenderer():
    while True:
        if not (canStart):
            continue
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = getRoi(frame)
        filtered = filterImage(roi)
        ret, buffer = cv2.imencode('.jpg', filtered)
        imgBytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + imgBytes + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/webcam_image')
def webcam_feed():
    return Response(cameraRenderer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/filtered_image')
def filter_feed():
    return Response(filterRenderer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/output_image')
def output_feed():
    return Response(outputRenderer(), mimetype='multipart/x-mixed-replace; boundary=frame')


@ app.route('/start')
def start():
    global canStart
    if (len(outputImages) == 0):
        return redirect('/upload_page')
    canStart = True
    return render_template('start.html')


@app.route('/upload_page')
def upload_feed():
    return render_template('upload.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == ALLOWED_EXTENSION


@app.route("/upload_image", methods=['POST'])
def upload():
    global totalImages, outputImages, outputImageEdited
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        i = 0
        for file in files:
            if file and allowed_file(file.filename):
                i += 1
                filename = 'img'+str(i)+'.jpg'
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    totalImages = len(os.listdir(uploadsPath))
    outputImages = []
    for i in range(1, totalImages+1):
        outputImages.append(cv2.imread(uploadsPath+"/img"+str(i)+".jpg"))
        outputImageEdited = outputImages[0].copy()

    flash('File(s) successfully uploaded')
    return redirect('/start')


@app.route("/stop")
def stop():
    global canStart, outputImages
    outputImages = []
    canStart = False
    shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)
    return redirect('/')


def startPredict():
    global predicted
    while True:
        if not canStart:
            continue
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = getRoi(frame)
        filtered = filterImage(roi)
        result = loaded_model.predict(filtered.reshape(1, 64, 64, 1))
        print(result)
        prediction = {'0': result[0][0],
                      '1': result[0][1],
                      '2': result[0][2],
                      '3': result[0][3],
                      '4': result[0][4],
                      '5': result[0][5]}
        prediction = sorted(prediction.items(),
                            key=operator.itemgetter(1), reverse=True)
        predicted = prediction[0][0]
        time.sleep(1)
        cv2.waitKey(10)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    modelThread = threading.Thread(target=startPredict)
    webThread = threading.Thread(target=app.run)
    modelThread.start()
    webThread.start()
