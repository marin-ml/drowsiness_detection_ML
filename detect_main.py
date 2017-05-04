
import cv2
from predict import face_classify
import time
import pygame

""" ----------------- Create the haar cascade ---------------- """
faceCascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('xml/haarcascade_eye_tree_eyeglasses.xml')
mouthCascade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')

""" -------------- Create the CNN classifier class ------------ """
my_class = face_classify()

""" ------------------ Color and variable define -------------- """
color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_white = (255, 255, 255)
color_black = (0, 0, 0)

state = ['Close', 'Open']
cap = cv2.VideoCapture(0)
close_eye_start = time.time()
open_mouth_start = time.time()
mouth_open_time = []
time_ind = 0
f_close = True

""" ---------------- camera read and process loop ------------- """
while True:
    # ------------ read image from camera and convert to gray -------------
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------- get the camera video size ----------------------
    height, width = image.shape[:2]
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_black, 54)
    cv2.rectangle(image, (30, height - 30), (450, height - 30), color_white, 50)
    cv2.rectangle(image, (510, height - 30), (width - 30, height - 30), color_black, 54)
    cv2.rectangle(image, (510, height - 30), (width - 30, height - 30), color_white, 50)

    # ----- Detect face, mouth and eyes in the image using haar cascade ----
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=2)
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=2)
    mouth = mouthCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=2)

    # -------------------- Calibrate the face data ------------------------
    if faces.__len__() > 0:
        face_data = [faces[0]]
    else:
        face_data = [[0, 0, 0, 0]]

    # -------------------- Calibrate the eyes data ------------------------
    eye_data = []
    for (x, y, w, h) in eyes:
        if x > face_data[0][0] and (x + w) < (face_data[0][0] + face_data[0][2]):           # out of x direction
            if y > face_data[0][1] and (y + h) < (face_data[0][1] + face_data[0][3]):       # out of y direction
                if (y + h/2) < (face_data[0][1] + face_data[0][3]/2):                       # below face
                    eye_data.append([x, y, w, h])

    # -------------------- Calibrate the mouth data ------------------------
    mouth_data = []
    for (x, y, w, h) in mouth:
        if x > face_data[0][0] and x + w < face_data[0][0] + face_data[0][2]:               # out of x direction
            if y + h/2 < face_data[0][1] + face_data[0][3]:                                 # out of y direction
                if y > face_data[0][1] + face_data[0][3]/2:
                    if y < face_data[0][1] + face_data[0][3]:
                        if y + h > face_data[0][1] + face_data[0][3] * 0.8:
                            mouth_data.append([x, y, w, h])

    # ----------------- get mouth image data for deep learning --------------
    ret_mouth = 1
    for (x, y, w, h) in mouth_data:
        img_mouth = gray[y:y + h, x:x + w]
        ret_mouth = my_class.classify(img_mouth, 'mouth')

    ret_eye = [0, 0]
    eye_ind = 0
    ret_eyes = 1
    for (x, y, w, h) in eye_data:
        img_eye = gray[y:y + h, x:x + w]
        ret_eye[eye_ind] = my_class.classify(img_eye, 'eye')
        eye_ind += 1
        if eye_ind > 1:
            break

    ret_eyes = ret_eye[0] and ret_eye[1]

    # -------------------------- deciding drowsiness -------------------------
    if ret_eyes == 1:
        close_eye_start = time.time()
        close_eye_time = 0
    else:
        close_eye_time = time.time() - close_eye_start
    # print close_eye_time

    if ret_mouth == 0:
        open_mouth_start = time.time()
        open_mouth_time = 0
        f_close = True
    else:
        open_mouth_time = max(time.time() - open_mouth_start - 0.5, 0)
        if open_mouth_time > 0 and f_close:
            mouth_open_time.append(time.time())
            f_close = False

    for time_ind in range(len(mouth_open_time)):
        if mouth_open_time[time_ind] > time.time() - 60:
            break
    # print len(mouth_open_time) - time_ind

    # ------------------ print the result text on the image ------------------
    cv2.putText(image, "eye:" + str(int(close_eye_time)) + ',' + state[ret_eyes],
                (20, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_blue, 2)
    cv2.putText(image, "mouth:" + str(len(mouth_open_time) - time_ind) + ',' + state[ret_mouth],
                (230, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_blue, 2)
    if close_eye_time > 20 or len(mouth_open_time) - time_ind > 10:
        cv2.putText(image, "Snooze", (505, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_red, 2)
        pygame.init()
        pygame.mixer.music.load('Alarm.wav')
        pygame.mixer.music.play()
    else:
        cv2.putText(image, "Ok", (540, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_red, 2)

    # # ------------------- Draw a rectangle around the faces -------------------
    # for (x, y, w, h) in eye_data:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # for (x, y, w, h) in mouth_data:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    # for (x, y, w, h) in face_data:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # -------------------- display the result image ---------------------------
    cv2.imshow("Faces found", image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

""" --------------------- device release and free --------------------------- """
cap.release()
cv2.destroyAllWindows()
