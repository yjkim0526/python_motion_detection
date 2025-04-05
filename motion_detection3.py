import cv2
import time

# 첫 번째 프레임 읽기
cap = cv2.VideoCapture("rtsp://210.99.70.120:1935/live/cctv002.stream")
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

cnt = 0
while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    cnt += 1
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

    # 프레임 간 차이 계산
    delta_frame = cv2.absdiff(frame1_gray, frame2_gray)
    _, thresh = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)
    
    # 모션이 감지되면 알림
    motion_sensitivity = 1000
    if cv2.countNonZero(thresh) > motion_sensitivity:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Motion detected! [{str(cnt)}] - Time: {current_time}")

    frame1_gray = frame2_gray
    cv2.imshow("Video", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()