# pushup.py
import cv2
import joblib
import math
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import mediapipe as mp
import time
from ultralytics import YOLO

label = ['정자세', '오자세']
model = YOLO('yolov5/yolo11n.pt')
pushup_model = joblib.load('C:/ai5/본프로젝트/메인/_data/운동별 모델/crunch.pkl')
font_path = "C:/ai5/본프로젝트/메인/_data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 20)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def crunch_info(st):
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>크런치</h1>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
                <div style="
                    background-color: #fff; 
                    padding: 30px; 
                    border-radius: 15px; 
                    border: 1px solid #ddd; 
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
                    max-width: 600px; 
                    margin: auto;
                    font-family: 'Roboto', Arial, sans-serif;
                ">
                    <div>
                        <h2 style="font-size: 20px; color: #333; margin: 0; text-align: center;">크런치</h3>
                        <h3 style="font-size: 18px; color: #333; margin: 0;">1. 카메라 설정 안내</h3>
                        <ul style="font-size: 14px; color: #555; line-height: 1.8; padding-left: 20px; margin-top: 10px;">
                            <li>휴대폰을 바닥에 세워두시고, <b>좌측</b> 또는 <b>우측 전신</b>이 화면에 잘 나오도록 카메라를 조정해주세요.</li>
                            <li>주변 환경이 밝고, 촬영에 방해되지 않도록 정리해주세요.</li>
                        </ul>
                    </div>
                    <div style="
                        border-top: 1px solid #eee; 
                        padding-top: 20px; 
                        margin-top: 20px;
                    ">
                        <h3 style="font-size: 18px; color: #333; margin: 0;">2. 운동 시작</h3>
                        <ul style="font-size: 14px; color: #555; line-height: 1.8; padding-left: 20px; margin-top: 10px;">
                            <li>준비가 완료되셨다면, '다음' 버튼을 눌러주세요!</li>
                            <li>운동이 바로 시작됩니다!</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.image("C:/ai5/본프로젝트/메인/_data/크런치.png", use_container_width=True)
    
        if st.button("다음"):
            st.session_state.current_exercise_state = "camera"
            st.rerun()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def calculate_angle(a, b, c):
    if hasattr(a, 'x'):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
    else:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def flip_landmarks(landmarks):
    flipped = [None] * len(landmarks)
    swap_indices = {0: 0, 1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14,
                   14: 13, 15: 16, 16: 15, 17: 18, 18: 17, 19: 20, 20: 19, 21: 22, 22: 21, 23: 24, 24: 23, 25: 26,
                   26: 25, 27: 28, 28: 27, 29: 30, 30: 29, 31: 32, 32: 31}
    for idx, lm in enumerate(landmarks):
        # 좌우 대칭 위치 결정
        new_idx = swap_indices.get(idx, idx)  # 좌우가 바뀌는 인덱스가 없으면 그대로 유지
        flipped[new_idx] = type(lm)(
            x=1.0 - lm.x,  # X 좌표 반전
            y=lm.y,  # Y, Z 좌표는 그대로
            z=lm.z,
            visibility=lm.visibility
        )
    return flipped

def run_crunch_session(st, YOLO, info):
    # 초기 변수 설정
    total_history = []  # 전체 프레임의 예측 결과 저장
    correct_threshold = 0.5  # 정자세 비율 기준 (70%)
    frame_skip = 4
    frame_count = 0
    crunch_state = "내려감"  # 초기 상태
    crunch_count = 0  # 카운트 초기화
    wrong_crunch_count = 0
    
    # 세트 및 횟수 설정
    target_reps = info["횟수"]  # 목표 횟수
    target_sets = info["세트"]  # 목표 세트
    completed_sets = 0  # 완료된 세트 수

    # 결과 저장용 딕셔너리 초기화
    results_dict = {
        "정자세 횟수": 0,
        "오자세 횟수": 0,
        "무릎이 발끝을 넘지 않도록 하세요.": 0,
        "몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다.": 0,
        "엉덩이와 허리를 일직선으로 유지하세요.": 0
    }

    # MediaPipe Pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 스트림릿 화면 초기화
    st.markdown("<h1 style='text-align: center;'>크런치</h1>", unsafe_allow_html=True)

    # 스트리밍 비디오 피드 설정
    video_url = "C:/ai5/본프로젝트/메인/영상/크런치/사영1.mp4"
    ip_webcam_url = "http://192.168.0.98:8080/video"    

    # 채팅 메시지 창
    frame_placeholder = st.empty()
    chat_placeholder = st.empty()
    count_placeholder = st.empty()
    set_placeholder = st.empty()  # 세트 표시용

    # 비디오 스트리밍 처리
    cap = cv2.VideoCapture(video_url)  # 동영상 입력
    # cap = cv2.VideoCapture(0)       # 웹캠
    # cap = cv2.VideoCapture(ip_webcam_url) # 폰 카메라
    # cap.set(cv2.CAP_PROP_FPS, 15)

    if "feedback_display_start_time" not in st.session_state:
        st.session_state.feedback_display_start_time = None 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # 프레임 건너뛰기
            continue
        
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # YOLO 모델로 사람 바운딩 박스 생성
        results = model(frame)
        best_box = None
        best_confidence = 0.0

        for result in results:
            for box in result.boxes:
                class_id = box.cls[0]
                confidence = box.conf[0]
                if int(class_id) == 0 and confidence > best_confidence:
                    best_box = box
                    best_confidence = confidence

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]

            with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
                results_pose = pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                landmarks_data = []

                if results_pose.pose_landmarks:
                    try:
                        landmarks = results_pose.pose_landmarks.landmark
                        
                        # 얼굴 방향 확인 (왼쪽 또는 오른쪽)
                        # nose_x = landmarks[0].x  # 코의 X 좌표
                        # shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2  # 어깨 중심의 X 좌표
                        # is_left_facing = nose_x > shoulder_center_x
                        
                        # # 왼쪽을 바라보는 경우 랜드마크 좌우 대칭 변환
                        # if is_left_facing:
                        #     landmarks = flip_landmarks(landmarks)  # 좌우 대칭 변환
                        
                        # 랜드마크 데이터 생성
                        landmarks_data = []
                        for landmark in landmarks:
                            landmarks_data.extend([landmark.x, landmark.y, landmark.z])

                        # 허리 좌표 계산
                        waist_x = (landmarks[23].x + landmarks[24].x) / 2
                        waist_y = (landmarks[23].y + landmarks[24].y) / 2
                        waist_z = (landmarks[23].z + landmarks[24].z) / 2
                        landmarks_data.extend([waist_x, waist_y, waist_z])

                        # 각도 계산
                        left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                        right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                        left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                        right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
                        landmarks_data.extend([
                            left_knee_angle, right_knee_angle,
                            left_hip_angle, right_hip_angle,
                        ])

                        # 자세 예측
                        prediction = pushup_model.predict([landmarks_data])
                        prediction_text = label[int(prediction[0])]

                        total_history.append(prediction_text == "정자세")
                        correct_ratio = sum(total_history) / len(total_history)
                        
                        # 어깨, 엉덩이, 무릎 좌표 추출
                        shoulder = (landmarks[11].x, landmarks[11].y, landmarks[11].z)
                        hip = (landmarks[23].x, landmarks[23].y, landmarks[23].z)
                        knee = (landmarks[25].x, landmarks[25].y, landmarks[25].z)
                        
                        # 크런치 동작 판별
                        if crunch_state == "올라감":
                            angle = calculate_angle(shoulder, hip, knee)
                            print(f"현재 각도: {angle:.2f}")
                            if angle > 94 and correct_ratio >= correct_threshold:  # 각도가 90도 이하로 내려갔을 때
                                print("각도가 충분히 내려갔습니다!")
                                crunch_state = "내려감"  # 상태 전환
                                print(crunch_state)
                                total_history=[]

                        elif crunch_state == "내려감":
                            angle = calculate_angle(shoulder, hip, knee)
                            print(f"현재 각도: {angle:.2f}")
                            if angle < 90 :
                                if correct_ratio >= correct_threshold:  # 각도가 충분히 올라갔을 때
                                    print("각도가 충분히 올라갔습니다!")
                                    crunch_count += 1  # 카운트 증가
                                    print(f"크런치 동작 완료! 현재 카운트: {crunch_count}")
                                    crunch_state = "올라감"  # 상태 전환
                                    print(crunch_state)
                                    
                                    if crunch_count == target_reps:
                                        completed_sets += 1
                                        crunch_count = 0
                                        
                                        if completed_sets == target_sets:
                                            st.success("모든 세트를 완료했습니다! 수고하셨습니다!")
                                            cap.release()
                                            cv2.destroyAllWindows()
                                            results_dict["정자세 횟수"] = completed_sets * target_reps
                                            return results_dict
                                else:
                                    wrong_crunch_count += 1
                                    crunch_state = "올라감"
                        
                        # "오자세"인 경우 피드백 생성 및 출력
                        if prediction_text == "오자세":
                            feedback_messages = []

                            # 무릎-발끝 정렬 피드백
                            knee_to_toe_distance = abs(landmarks[25].x - landmarks[27].x)  # 왼쪽 무릎과 발의 X좌표 차이
                            if knee_to_toe_distance > 0.15:  # 무릎이 발끝을 지나쳤을 때
                                results_dict["무릎이 발끝을 넘지 않도록 하세요."] += 1
                                feedback_messages.append("무릎이 발끝을 넘지 않도록 하세요.")

                            # 몸의 균형 피드백
                            body_tilt_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])  # 상체 기울기
                            if body_tilt_angle < 75 or body_tilt_angle > 105:  # 몸이 너무 기울어졌을 때
                                results_dict["몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다."] += 1
                                feedback_messages.append("몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다.")

                            # 엉덩이와 허리 위치 피드백
                            hip_knee_alignment = calculate_angle(landmarks[23], landmarks[25], landmarks[27])  # 엉덩이-무릎-발목 각도
                            if hip_knee_alignment < 90 or hip_knee_alignment > 120:  # 엉덩이가 너무 낮거나 높을 때
                                results_dict["엉덩이와 허리를 일직선으로 유지하세요."] += 1
                                feedback_messages.append("엉덩이와 허리를 일직선으로 유지하세요.")
                                                    
                            # 피드백 출력
                            if feedback_messages:
                                feedback_text = "<br>".join(feedback_messages)
                                chat_placeholder.markdown(feedback_text, unsafe_allow_html=True)
                                st.session_state.feedback_display_start_time = time.time()  # 피드백 표시 시작 시간 기록
                        else:
                            # "오자세"가 아닐 경우 피드백 내용을 5초 후에 지웁니다.
                            if st.session_state.feedback_display_start_time and time.time() - st.session_state.feedback_display_start_time > 5:
                                chat_placeholder.markdown("", unsafe_allow_html=True)
                                st.session_state.feedback_display_start_time = None  # 표시 시간 초기화
                        
                        #  # 출력 텍스트 생성
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # draw = ImageDraw.Draw(pil_img)
                        # #draw.text((x1, y1 - 80), f"State: {current_state}", font=font, fill=(255, 255, 0))
                        # draw.text((x1, y1 - 40), f"Count: {crunch_count} | Correct Ratio: {correct_ratio:.2f} | {prediction_text}", font=font, fill=(255, 0, 0))
                        # frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        # mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # 횟수 표시
                        count_placeholder.info(f"크런치 횟수 : {crunch_count}/{target_reps}    크런치 오자세 횟수 : {wrong_crunch_count}     세트 : {completed_sets}/{target_sets}")
                        # count_placeholder.info(f"크런치 횟수 : {crunch_count}/{target_reps}   세트 : {completed_sets}/{target_sets}")

                    except IndexError:
                        print("IndexError: 랜드마크 데이터 추출 오류")

        # 회전된 프레임을 표시
        frame_placeholder.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    results_dict["정자세 횟수"] = completed_sets * target_reps
    results_dict['오자세 횟수'] = wrong_crunch_count
    
    total_feedback = results_dict['무릎이 발끝을 넘지 않도록 하세요.'] + results_dict['몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다.'] + results_dict['엉덩이와 허리를 일직선으로 유지하세요.']
    results_dict['무릎이 발끝을 넘지 않도록 하세요.'] = round(results_dict['무릎이 발끝을 넘지 않도록 하세요.'] / total_feedback, 5)
    results_dict['몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다.'] = round(results_dict['몸의 균형을 유지하세요. 상체가 너무 기울어졌습니다.'] / total_feedback , 5)
    results_dict['엉덩이와 허리를 일직선으로 유지하세요.'] = round(results_dict['엉덩이와 허리를 일직선으로 유지하세요.'] / total_feedback, 5)
    

    return results_dict

