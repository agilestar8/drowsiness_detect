import cv2
import numpy as np
import dlib
from keras.models import load_model
from scipy.spatial import distance as dist  # 유클리드 거리 계산용
from imutils import face_utils

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 16
COUNTER = 0
yawn_cnt = 0
drowsy_time = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

model = load_model('drowiness_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 캐스캐이드 분류 모델 사용
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # dlib 얼굴마크 데이터
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)  # 비디오 객체
cap.set(3, 640)  # 너비 설정
cap.set(4, 480)  # 높이 설정

while True:
    ret, frame = cap.read()     # 비디오를 프레임 단위로 읽음, 정상작동 시 ret == True, frame == 읽은 비디오프레임
    frame = cv2.flip(frame, 1)  # 화면 좌우대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # 프레임을 흑백(gray)로 변환
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)    # 얼굴검출, 검출 성공 시 x,y,w,h 좌표 리턴

    if len(faces):
        for (x, y, w, h) in faces:
            ### yawn detect
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴좌표에 사각형 그리기
            roi_color = img[y:y + h, x:x + w]
            resized_array = cv2.resize(roi_color, (145, 145))
            resized_array2 = resized_array.reshape(-1,145,145,3)
            condition = model.predict(resized_array2)   # 사용자 현재상태 판정
            condition2 = np.argmax(condition, axis=1)
            # cv2.imwrite("face.jpg", roi_color) # 이미지 확인

            if condition2 == 0:
                cv2.putText(frame, "yawned", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                yawn_cnt += 1
                # 프레임 당 하품시간
                if yawn_cnt > 20:
                    cv2.putText(frame, "you are drowsy", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255),2)
                    drowsy_time += 1

            elif condition2 == 1:
                cv2.putText(frame, "Normal", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                yawn_cnt = 0
                drowsy_time = 0

            cv2.putText(frame, "drowsy_time : {:.2f}".format(drowsy_time), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ### EAR part
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

            # EAR이 0.3을 넘기면
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 20 프레임이상 EAR이 기준치 미달 시
                    cv2.putText(frame, "DROWSINESS ALERT!", (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    drowsy_time += 1
            else:
                COUNTER = 0
                drowsy_time = 0

            cv2.putText(frame, "EAR: {:.3f}".format(ear), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if drowsy_time > 40:
                cv2.putText(frame, "Please take a break ", (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('CAM1', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Esc 키를 누르면 종료
        break

cap.release()           # 비디오 객체 해제
cv2.destroyAllWindows() # 윈도우 제거