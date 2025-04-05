import cv2
import numpy as np
from datetime import datetime
import os

# 녹화 파일을 저장할 디렉토리 생성
if not os.path.exists('recordings'):
    os.makedirs('recordings')

# 웹캠 연결
cap = cv2.VideoCapture(0)

# 프레임 크기 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# 초기 설정
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# 모션 감지 민감도 (높을수록 덜 민감)
motion_threshold = 10000
is_recording = False
record_counter = 0

while True:
    # 움직임 감지
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 움직임이 감지되면 녹화 시작
    if motion_detected and not is_recording:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'recordings/motion_{current_time}.avi'
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
        is_recording = True
        record_counter = 0
    
    # 움직임이 감지되지 않으면 카운터 증가
    if not motion_detected and is_recording:
        record_counter += 1
        # 3초(90프레임) 동안 움직임이 없으면 녹화 중지
        if record_counter > 90:
            is_recording = False
            if out is not None:
                out.release()
                out = None
    
    # 녹화 중이면 프레임 저장
    if is_recording and out is not None:
        out.write(frame1)
        # 녹화 중임을 표시
        cv2.circle(frame1, (30, 30), 10, (0, 0, 255), -1)
    
    # 상태 텍스트 표시
    status = "Motion Detected" if motion_detected else "No Motion"
    cv2.putText(frame1, f"Status: {status}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if is_recording:
        cv2.putText(frame1, "Recording...", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 결과 화면 표시
    cv2.imshow("Motion Detection", frame1)
    
    # 프레임 업데이트
    frame1 = frame2
    ret, frame2 = cap.read()
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # 's' 키를 누르면 민감도 조절
    elif cv2.waitKey(1) & 0xFF == ord('s'):
        motion_threshold = int(input("새로운 민감도 값을 입력하세요 (기본값: 10000): "))

# 자원 해제
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows() 