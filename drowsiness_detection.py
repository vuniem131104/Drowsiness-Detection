import dlib 
import cv2
import numpy as np
from imutils import face_utils
import playsound
from  threading import Thread

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
	A = np.linalg.norm(eye[1] - eye[5])
	B = np.linalg.norm(eye[2] - eye[4])
	C = np.linalg.norm(eye[0] - eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
video = cv2.VideoCapture(0)
ret, frame = video.read()
h, w = frame.shape[:2]
# out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
THRESHOLD = 0.3
NUM_FRAMES = 48
TOTAL, COUNTER = 0, 0
ALARM_ON = False
EAR = 0
while ret:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        faces = detector(gray_frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)
            landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
            for i in range(rStart, rEnd):
                cv2.circle(frame, tuple(landmarks_array[i]), 1, (0, 255, 0), -1)
            for i in range(lStart, lEnd):
                cv2.circle(frame, tuple(landmarks_array[i]), 1, (0, 255, 0), -1)
            EAR = round((eye_aspect_ratio(landmarks_array[rStart:rEnd]) + eye_aspect_ratio(landmarks_array[lStart:lEnd])) / 2, 2)
        if len(faces) > 0:
            cv2.putText(frame, f'EAR: {str(EAR)}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if EAR < THRESHOLD:
                COUNTER += 1
                if COUNTER >= NUM_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=sound_alarm,
                            args=('alert.mp3',))
                        t.deamon = True
                        t.start()     
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                ALARM_ON = False 
                COUNTER = 0
        cv2.imshow('Video', cv2.resize(frame, (int(w * 1.5), int(h * 1.5))))
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break 
    else:
        break
    ret, frame = video.read()
    
# out.release()
video.release()
cv2.destroyAllWindows()