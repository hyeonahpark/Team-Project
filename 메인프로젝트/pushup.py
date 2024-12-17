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
pushup_model = joblib.load('C:/ai5/본프로젝트/메인/_data/운동별 모델/pushup.pkl')
font_path = "C:/ai5/본프로젝트/메인/_data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 20)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def pushup_info(st):
    st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI 홈트레이너</h1>", unsafe_allow_html=True)

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
                        <h2 style="font-size: 20px; color: #333; margin: 0; text-align: center;">푸시업</h3>
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
        st.image("C:/ai5/본프로젝트/메인/_data/푸시업.png", use_container_width=True)
    
        if st.button("다음"):
            st.session_state.current_exercise_state = "camera"
            st.rerun()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def calculate_angle(point_a, point_b, point_c):
    ab = (point_b.x - point_a.x, point_b.y - point_a.y, point_b.z - point_a.z)
    bc = (point_c.x - point_b.x, point_c.y - point_b.y, point_c.z - point_b.z)
    dot_product = sum(a * b for a, b in zip(ab, bc))
    magnitude_ab = math.sqrt(sum(a**2 for a in ab))
    magnitude_bc = math.sqrt(sum(b**2 for b in bc)) 
    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0.0
    return math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))


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

def run_pushup_session(st, YOLO, info):
    # 초기 변수 설정
    pushup_count = 0  # 푸시업 정자세 개수
    pushup_wrong_count = 0  # 푸시업 오자세 개수
    down_position = False  # 몸이 아래로 내려간 상태 여부
    total_history = []  # 전체 프레임의 예측 결과 저장
    correct_threshold = 0.5  # 정자세 비율 기준 (70%)
    current_state = "대기"  # 초기 상태는 "대기"
    
    # 세트 및 횟수 설정
    target_reps = info["횟수"]  # 목표 횟수
    target_sets = info["세트"]  # 목표 세트
    completed_sets = 0  # 완료된 세트 수

    frame_skip = 4  # 1 프레임 건너뛰기
    frame_count = 0  # 프레임 카운터 초기화

    # 결과 저장용 딕셔너리 초기화
    results_dict = {
        "정자세 횟수": 0,
        "오자세 횟수": 0,
        "목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.": 0,
        "푸시업을 더 깊게 하세요.": 0,
        "고개가 정면을 향하게 해주세요.": 0
    }

    # MediaPipe Pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 스트림릿 화면 초기화
    st.markdown("<h1 style='text-align: center;'>푸시업</h1>", unsafe_allow_html=True)

    # 스트리밍 비디오 피드 설정
    video_url = "C:/ai5/본프로젝트/메인/영상/푸시업/사영2.mp4"
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
    cap.set(cv2.CAP_PROP_FPS, 15)

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
                        # Mediapipe 랜드마크 추출
                        landmarks = results_pose.pose_landmarks.landmark
                        
                        # 얼굴 방향 확인 (왼쪽 또는 오른쪽)
                        nose_x = landmarks[0].x  # 코의 X 좌표
                        shoulder_center_x = (landmarks[11].x + landmarks[12].x) / 2  # 어깨 중심의 X 좌표
                        is_left_facing = nose_x < shoulder_center_x

                        # 왼쪽을 바라보는 경우 랜드마크 좌우 대칭 변환
                        if is_left_facing:
                            landmarks = flip_landmarks(landmarks)  # 좌우 대칭 변환

                        # 이후 landmarks 데이터를 사용하여 예측 모델에 넣을 데이터 구성
                        landmarks_data = []

                        # 주요 랜드마크 추출
                        for idx in [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]:
                            landmarks_data.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

                        # 추가 계산
                        neck_x = (landmarks[11].x + landmarks[12].x) / 2
                        neck_y = (landmarks[11].y + landmarks[12].y) / 2
                        neck_z = (landmarks[11].z + landmarks[12].z) / 2
                        landmarks_data.extend([neck_x, neck_y, neck_z])

                        left_palm_x = (landmarks[19].x + landmarks[21].x + landmarks[17].x + landmarks[15].x) / 4
                        left_palm_y = (landmarks[19].y + landmarks[21].y + landmarks[17].y + landmarks[15].y) / 4
                        left_palm_z = (landmarks[19].z + landmarks[21].z + landmarks[17].z + landmarks[15].z) / 4
                        right_palm_x = (landmarks[20].x + landmarks[22].x + landmarks[18].x + landmarks[16].x) / 4
                        right_palm_y = (landmarks[20].y + landmarks[22].y + landmarks[18].y + landmarks[16].y) / 4
                        right_palm_z = (landmarks[20].z + landmarks[22].z + landmarks[18].z + landmarks[16].z) / 4
                        landmarks_data.extend([left_palm_x, left_palm_y, left_palm_z, right_palm_x, right_palm_y, right_palm_z])

                        shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                        shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                        shoulder_mid_z = (landmarks[11].z + landmarks[12].z) / 2
                        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                        hip_mid_z = (landmarks[23].z + landmarks[24].z) / 2

                        vector_x = hip_mid_x - shoulder_mid_x
                        vector_y = hip_mid_y - shoulder_mid_y
                        vector_z = hip_mid_z - shoulder_mid_z

                        back_x = shoulder_mid_x + (1/3) * vector_x
                        back_y = shoulder_mid_y + (1/3) * vector_y
                        back_z = shoulder_mid_z + (1/3) * vector_z
                        waist_x = shoulder_mid_x + (2/3) * vector_x
                        waist_y = shoulder_mid_y + (2/3) * vector_y
                        waist_z = shoulder_mid_z + (2/3) * vector_z
                        landmarks_data.extend([back_x, back_y, back_z, waist_x, waist_y, waist_z])

                        # 거리 및 각도 계산 추가
                        shoulder_distance = calculate_distance(landmarks[11], landmarks[12])
                        left_arm_length = calculate_distance(landmarks[11], landmarks[13]) + calculate_distance(landmarks[13], landmarks[15])
                        right_arm_length = calculate_distance(landmarks[12], landmarks[14]) + calculate_distance(landmarks[14], landmarks[16])
                        left_leg_length = calculate_distance(landmarks[23], landmarks[25]) + calculate_distance(landmarks[25], landmarks[27])
                        right_leg_length = calculate_distance(landmarks[24], landmarks[26]) + calculate_distance(landmarks[26], landmarks[28])
                        leg_distance = calculate_distance(landmarks[27], landmarks[28])
                        left_arm_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                        right_arm_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                        body_incline_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[27])

                        landmarks_data.extend([
                            shoulder_distance, left_arm_length, right_arm_length,
                            left_leg_length, right_leg_length, leg_distance,
                            left_arm_angle, right_arm_angle, body_incline_angle
                        ])
                
                        # 거리 및 각도 계산 추가
                        left_arm_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                        right_arm_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                        avg_elbow_angle = min(left_arm_angle, right_arm_angle)

                        # 자세 예측
                        prediction = pushup_model.predict([landmarks_data])
                        prediction_text = label[int(prediction[0])]

                        # 전체 기록에 추가
                        total_history.append(prediction_text == "정자세")

                        # 정자세 비율 계산
                        correct_ratio = sum(total_history) / len(total_history)
                        print(f"Current State: {current_state}, Down Position: {down_position}")
                        print(f"Left Arm Angle: {left_arm_angle}, Right Arm Angle: {right_arm_angle}")
                        print(f"Average Elbow Angle: {avg_elbow_angle}")
                        print(f"Correct Ratio: {correct_ratio:.2f}")

                        # 푸시업 상태 판별 및 카운트 로직
                        if avg_elbow_angle < 20:  # "업" 상태
                            if current_state == "다운":
                                current_state = "업"
                                down_position = False
                                if correct_ratio >= correct_threshold:
                                    pushup_count += 1
                                    # correct_ratio를 새로 산출하기 위해 기록 초기화
                                    total_history = []  # 새로운 푸시업을 기준으로 기록 초기화
                                    # 목표 횟수를 달성한 경우
                                    if pushup_count == target_reps:
                                        completed_sets += 1
                                        pushup_count = 0  # 횟수 초기화

                                        if completed_sets == target_sets:  # 모든 세트 완료
                                            st.success("모든 세트를 완료했습니다! 수고하셨습니다!")
                                            cap.release()
                                            cv2.destroyAllWindows()
                                            results_dict["정자세 횟수"] = completed_sets * target_reps
                                            return results_dict
                                else:
                                    pushup_wrong_count +=1
                        elif avg_elbow_angle >= 40:  # "다운" 상태
                            current_state = "다운"

                        # "오자세"인 경우 피드백 생성 및 출력
                        if prediction_text == "오자세" :
                            feedback_messages = []
                            
                            # 어깨와 손 위치 불일치 피드백
                            # shoulder_wrist_distance = calculate_distance(landmarks[11], landmarks[15])
                            # if shoulder_wrist_distance > 0.15:  # 어깨와 손의 위치가 일치하지 않을 때
                            #     feedback_messages.append("어깨와 손의 위치를 일치하게 해주세요.")
                            
                            # 목-허리-엉덩이-다리 일직선 피드백
                            body_alignment_angle = calculate_angle(landmarks[0], landmarks[23], landmarks[27])
                            if body_alignment_angle < 165:  # 일직선이 아닐 때
                                results_dict["목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요."] += 1
                                feedback_messages.append("목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.")
                            
                            # 깊이 부족 피드백
                            if avg_elbow_angle > 45:  # 팔꿈치 각도가 충분히 접히지 않았을 때
                                results_dict["푸시업을 더 깊게 하세요."] += 1
                                feedback_messages.append("푸시업을 더 깊게 하세요.")
                            
                            # 고개 정면 피드백
                            if abs(nose_x - shoulder_center_x) > 0.1:  # 고개가 정면을 향하고 있지 않을 때
                                results_dict["고개가 정면을 향하게 해주세요."] += 1
                                feedback_messages.append("고개가 정면을 향하게 해주세요.")
                            
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
                        # draw.text((x1, y1 - 40), f"Count: {pushup_count} | Correct Ratio: {correct_ratio:.2f} | {prediction_text}", font=font, fill=(255, 0, 0))
                        # frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        # mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # 횟수 표시
                        count_placeholder.info(f"푸시업 횟수 : {pushup_count}/{target_reps}    푸시업 오자세 횟수 : {pushup_wrong_count}   세트 : {completed_sets}/{target_sets}")
                        # count_placeholder.info(f"푸시업 횟수 : {pushup_count}/{target_reps}   세트 : {completed_sets}/{target_sets}")

                    except IndexError:
                        print("IndexError: 랜드마크 데이터 추출 오류")

        # 회전된 프레임을 표시
        frame_placeholder.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    results_dict["정자세 횟수"] = completed_sets * target_reps
    results_dict['오자세 횟수'] = pushup_wrong_count
    
    total_feedback = results_dict['고개가 정면을 향하게 해주세요.'] + results_dict['목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.'] + results_dict['푸시업을 더 깊게 하세요.']
    results_dict['고개가 정면을 향하게 해주세요.'] = round(results_dict['고개가 정면을 향하게 해주세요.'] / total_feedback, 5)
    results_dict['목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.'] = round(results_dict['목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.'] / total_feedback , 5)
    results_dict['푸시업을 더 깊게 하세요.'] = round(results_dict['푸시업을 더 깊게 하세요.'] / total_feedback, 5)
    

    return results_dict

# {'정자세 횟수': 3, '오자세 횟수': 3, '목과 허리, 엉덩이, 다리가 일직선이 되도록 해주세요.': 0.4734, '푸시업을 더 깊게 하세요.': 0.1117, '고개가 정면을 향하게 해주세요.': 0.41489}