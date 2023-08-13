from flask import *
import cv2
import threading
import cv2
import numpy as np
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from ruamel import yaml
from ultralytics import YOLO
from utils import detect
from openvino.runtime import Core
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


app = Flask(__name__)
CORS(app)

outputFrame = None
lock = threading.Lock()

model = YOLO("yolov8n-pose")

data = yaml.safe_load(open('yolov8n/metadata.yaml'))
seg_model_path = 'yolov8n/yolov8n-seg.xml'
label_map = data['names']

core = Core()
seg_ov_model = core.read_model(seg_model_path)
device = "CPU"  # GPU
if device != "CPU":
    seg_ov_model.reshape({0: [1, 3, 640, 640]})
seg_compiled_model = core.compile_model(seg_ov_model, device)


counter  = 0
pose_count = [0, 0, 0, 0, 0, 0, 0]
filename = '6.mp4'
isCamera = False
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)



def gen_frams():
    global counter
    global pose_count
    global isCamera
    if(isCamera == True):
        # cap = cv2.VideoCapture(0)
        print('turn on cameraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        cap = cv2.VideoCapture('knee.mp4')
    else:
        cap = cv2.VideoCapture("./upload/"+filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    direct = ''
    x, y = 0, 0
    pre_x = 0
    pre_y = 0
    pre_direct =''
    Pose_name = ['Left Foot', 'Right Foot','Left Leg', 'Right Leg', 'Left Shoulder', 'Rigt Shoulder', 'Head']

    right_ankle =  (0, 0)
    left_ankle = (0, 0)
    left_knee = (0,0)
    right_knee = (0,0)
    left_shoulder = (0,0)
    right_shoulder = (0,0)
    head = (0,0)
    pos_x = []
    pos_y = []
    i = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            i+= 1
            if i%3:
                continue
            if frame is not None:
                frame_ = rescale_frame(frame, percent=100)
            pre_x = x
            pre_y = y
            pre_direct = direct
            # Recolor image to RGB
            image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)

            detections = detect(image, seg_compiled_model)[0]
            if len(detections['det']) == 0:
                continue
            dets = detections['det']
            segs = detections['segment']

            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                head = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                # elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                # wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                # hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            except:
                pass
            for i , seg_cnt in enumerate(segs):
                if dets[i][5] == 32:
                    (x,y),radius = cv2.minEnclosingCircle(seg_cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
                    # cv2.circle(image,center,radius,(0,255,0),2)
            print('Positio', int(y), int(pre_y))
            pos_x = [left_ankle[0]*width, right_ankle[0]*width, left_knee[0]*width, right_knee[0]*width, left_shoulder[0]*width, right_shoulder[0]*width, head[0]*width]
            pos_y = [left_ankle[1]*height, right_ankle[1]*height, left_knee[1]*height, right_knee[1]*height, left_shoulder[1]*height, right_shoulder[1]*height, head[1]*height]
            if(y!=0 and pre_y !=0):
                if(y-pre_y !=0):
                    if((y- pre_y)>2):
                        direct = "Down"
                    elif(y-pre_y < 0):
                        if(abs(y-pre_y) >2):
                            direct = 'Up'
                        if(direct == 'Up' and pre_direct == 'Down'):
                            counter += 1
                            print(counter)
                            diff = []

                            for i in range(7):
                                dis = int((pre_y + radius - pos_y[i])*(pre_y +radius - pos_y[i]) + (pre_x - pos_x[i])*(pre_x - pos_x[i]))
                                diff.append(dis) 
                            sort_diff = sorted(diff)

                            for j in range(7):
                                if(sort_diff[0] == diff[j]):
                                    index = j
                                    pose_count[j] += 1
                                    break
                            print(Pose_name[index])
                            print(diff)
                            print("aaaaaaaaaaaaaaaaaaa",left_ankle[0]*width, left_ankle[1]*height)     
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            #                         mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
            #                         mp_drawing.DrawingSpec(color=(203,17,17), thickness=2, circle_radius=2) 
            #                         )               
            ret , buffer = cv2.imencode('.jpg',image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield 'Hello '

@app.route('/video')
def video():
    return Response(gen_frams(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count', methods = ['GET'])
def count():
    return jsonify(result=counter)

@app.route('/leftfootcount', methods =['GET'])
def leftfootcount():
    return jsonify(result=pose_count)

# @app.route('/rightfootcount', methods =['GET'])
# def rightfootcount():
#     return jsonify(result=pose_count[1])

# @app.route('/leftkneecount', methods =['GET'])
# def leftkneecount():
#     return jsonify(result=pose_count[2])

# @app.route('/rightkneecount', methods =['GET'])
# def rightkneecount():
#     return jsonify(result=pose_count[3])

# @app.route('/leftshoudlercount', methods =['GET'])
# def leftshoudlercount():
#     return jsonify(result=pose_count[4])

# @app.route('/rightshouldercount', methods =['GET'])
# def rightshouldercount():
#     return jsonify(result=pose_count[5])

# @app.route('/headcount', methods =['GET'])
# def headcount():
#     return jsonify(result=pose_count[6])



@app.route('/counting')
def counting():
    global counter
    global pose_count
    counter = 0
    pose_count = [0,0,0,0,0,0,0]
    return render_template('count.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    global filename
    if request.method == 'POST':
        f = request.files['file']
        upload_path = 'upload'
        f.save(os.path.join(upload_path, secure_filename(f.filename)))
        filename = secure_filename(f.filename)
        print(filename)

    return redirect('./counting')

@app.route('/camera', methods =['GET', 'POST'])
def camera():
    global isCamera
    if request.method == 'POST':
        isCamera = False
    return redirect('./counting')


@app.route('/')
def index():
    return render_template('index.html')
    # return '1231322'

if __name__ == "__main__":
    app.run(debug=True) 