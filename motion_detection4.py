import cv2
import time
from ultralytics import YOLO
import numpy as np

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 가벼운 nano 모델 사용

# 비디오 캡처 설정
cap = cv2.VideoCapture("rtsp://210.99.70.120:1935/live/cctv002.stream")

# 화면 크기 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 객체 감지
    results = model(frame)
    
    # 감지된 객체 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 클래스 확인 (자동차, 트럭, 버스 등)
            if box.cls[0] in [2, 3, 5, 7]:  # COCO 데이터셋의 차량 관련 클래스
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 신뢰도 점수
                conf = float(box.conf[0])
                
                if conf > 0.5:  # 신뢰도 50% 이상만 처리
                    # 차량 영역 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 차량 영역에서 번호판 영역 추출 시도
                    vehicle_roi = frame[y1:y2, x1:x2]
                    if vehicle_roi.size > 0:
                        # 그레이스케일 변환
                        gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                        
                        # 이미지 이진화
                        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        
                        # 윤곽선 검출
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 500:  # 최소 영역 크기
                                # 윤곽선을 감싸는 직사각형
                                rect = cv2.minAreaRect(contour)
                                box_points = cv2.boxPoints(rect)
                                box_points = np.int0(box_points)
                                
                                # 번호판 후보 영역 표시 (빨간색)
                                cv2.drawContours(vehicle_roi, [box_points], 0, (0, 0, 255), 2)

    # 현재 시간 표시
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 