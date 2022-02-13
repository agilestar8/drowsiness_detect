import cv2
import numpy as np
# from keras.models import load_model
# model = load_model('drowiness_model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   # 캐스캐이드 분류 모델 사용
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)  # 비디오 객체
cap.set(3, 640)  # 너비 설정
cap.set(4, 480)  # 높이 설정

while True:
    ret, frame = cap.read()     # 비디오를 프레임 단위로 읽음, 정상작동 시 ret == True, frame == 읽은 비디오프레임
    frame = cv2.flip(frame, 1)  # 화면 좌우 대칭시키기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # 프레임을 흑백(gray)로 변환
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)    # 얼굴검출, 검출 성공 시 x,y,w,h 좌표 리턴

    if len(faces):
        for (x, y, w, h) in faces:
            face_img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # 캠에 프레임 좌표로 사각형 그리기
            cv2.putText(frame, "Detected", (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            faceROI = face_img[y:y + h, x:x + w]    # 검출한 face의 프레임
            cv2.imwrite("face.jpg", faceROI)        # 프레임의 이미지 확인

            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2, y2, w2, h2) in eyes:
                # eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                # radius = int(round((w2 + h2) * 0.25))
                # cv2.circle(frame, eye_center, radius, (255, 0, 0), 2)
                eye_img = cv2.rectangle(faceROI, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
                eyeROI = eye_img[y2:y2 + h2, x2:x2 + w2]
                resized_eye = cv2.resize(eyeROI, (145,145))
                cv2.imwrite("eye.jpg", resized_eye)    # 프레임의 이미지 확인

    cv2.imshow('result', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Esc 키를 누르면 종료
        break

cap.release()           # 비디오 객체 해제
cv2.destroyAllWindows() # 윈도우 제거