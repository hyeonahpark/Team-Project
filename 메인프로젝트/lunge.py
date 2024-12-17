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
# pushup_model = joblib.load('C:/ai5/본프로젝트/메인/_data/운동별 모델/lunge.pkl')
pushup_model = joblib.load('C:/ai5/본프로젝트/메인/_data/런지_유튭/모델_전이_3/cat_model_updated.pkl')
font_path = "C:/ai5/본프로젝트/메인/_data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 20)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def lunge_info(st):
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
                        <h2 style="font-size: 20px; color: #333; margin: 0; text-align: center;">런지</h3>
                        <h3 style="font-size: 18px; color: #333; margin: 0;">1. 카메라 설정 안내</h3>
                        <ul style="font-size: 14px; color: #555; line-height: 1.8; padding-left: 20px; margin-top: 10px;">
                            <li>휴대폰을 바닥에 세워두시고, <b>정면으로 전신</b>이 화면에 잘 나오도록 카메라를 조정해주세요.</li>
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
        st.image("C:/ai5/본프로젝트/메인/_data/사이드런지.png", use_container_width=True)
    
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

def run_lunge_session(st, YOLO, info):
    total_history = []  # 전체 프레임의 예측 결과 저장
    correct_threshold = 0.7  # 정자세 비율 기준 (70%)
    frame_skip = 4
    frame_count = 0
    lunge_count = 0  # 푸시업 개수
    wrong_count = 0
    state = "업"

    # 이전 프레임 무릎 좌표 초기화
    previous_left_hip_y = None
    previous_right_hip_y = None
    
    # 세트 및 횟수 설정
    target_reps = info["횟수"]  # 목표 횟수
    target_sets = info["세트"]  # 목표 세트
    completed_sets = 0  # 완료된 세트 수

    ###################### 결과 저장용 딕셔너리 초기화 ######################
    results_dict = {
        "정자세 횟수": 0,
        "오자세 횟수": 0,
        "무릎이 너무 앞으로 나갔어요. 조금만 뒤로": 0,
        "앞무릎을 90도로 만들어 주세요.": 0,
        "뒷무릎 너무 내려가지 마세요.": 0,
        "허리 쭉 펴고 상체를 바르게 해주세요.": 0
    }
    ##########################################################################

    # MediaPipe Pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 스트림릿 화면 초기화
    st.markdown("<h1 style='text-align: center;'>런지</h1>", unsafe_allow_html=True)

    # 스트리밍 비디오 피드 설정
    video_url = "C:/ai5/본프로젝트/메인/영상/런지/현아6.mp4"
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

                        # 이후 landmarks 데이터를 사용하여 예측 모델에 넣을 데이터 구성
                        landmarks_data = []
                        # indices_to_include = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

                        indices_to_include = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

                        for i in indices_to_include:
                            landmark = landmarks[i]
                            landmarks_data.extend([landmark.x, landmark.y, landmark.z])

                        # 허리 좌표 계산
                        waist_x = (landmarks[23].x + landmarks[24].x) / 2
                        waist_y = (landmarks[23].y + landmarks[24].y) / 2
                        waist_z = (landmarks[23].z + landmarks[24].z) / 2
                        landmarks_data.extend([waist_x, waist_y, waist_z])

                        # 거리 계산
                        foot_distance = calculate_distance(landmarks[29], landmarks[30])  # 좌우 발 간 거리
                        left_knee_to_foot_distance = calculate_distance(landmarks[25], landmarks[29])  # 왼쪽 무릎-발 거리
                        right_knee_to_foot_distance = calculate_distance(landmarks[26], landmarks[30])  # 오른쪽 무릎-발 거리

                        # 각도 계산
                        left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                        right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                        left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                        right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
                        torso_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[29])  # 상체와 바닥 간 각도

                        landmarks_data.extend([
                            left_knee_angle, right_knee_angle,
                            left_hip_angle, right_hip_angle,
                            foot_distance, left_knee_to_foot_distance,
                            right_knee_to_foot_distance, torso_angle
                        ])
                        
                        prediction = pushup_model.predict([landmarks_data])
                        prediction_text = label[int(prediction[0])]

                        # 전체 기록에 추가
                        total_history.append(prediction_text == "정자세")

                        # 정자세 비율 계산
                        correct_ratio = sum(total_history) / len(total_history)
                        print(f"Correct Ratio: {correct_ratio:.2f}")

                        left_hip_y = landmarks[23].y
                        right_hip_y = landmarks[24].y   
                        
                        print(f"left_hip_y : {left_hip_y}, right_hip_y : {right_hip_y}")
                        print(f"previous_left_hip_y: {previous_left_hip_y}, previous_right_hip_y: {previous_right_hip_y}")
                        print(f"lunge_count: {lunge_count}")
                        
                        # # 첫 프레임에서는 초기화만 수행
                        # if previous_left_hip_y is None or previous_right_hip_y is None:
                        #     previous_left_hip_y = left_hip_y
                        #     previous_right_hip_y = right_hip_y
                        #     continue

                        avg_knee_angle = min(left_knee_angle, right_knee_angle)
                        
                        if avg_knee_angle < 90:
                            if state == '다운': 
                                state = '업'
                                if correct_ratio >= correct_threshold:
                                    lunge_count += 1
                                    total_history = []
                                    if lunge_count == target_reps:
                                        completed_sets += 1
                                        lunge_count = 0
                                        
                                        if completed_sets == target_sets:
                                            st.success("모든 세트를 완료했습니다! 수고하셨습니다!")
                                            cap.release()
                                            cv2.destroyAllWindows()
                                            results_dict["정자세 횟수"] = completed_sets * target_reps
                                            return results_dict          
                                elif correct_ratio <= 0.5:
                                    wrong_count += 1
                                    total_history = []
                        elif avg_knee_angle >= 90 and avg_knee_angle<= 100:
                            state = '다운'
                            
                        print('각도 : ', avg_knee_angle)

                        # "오자세"인 경우 피드백 생성 및 출력
                        if prediction_text == "오자세" :
                            feedback_messages = []
                            
                            # 앞무릎 위치 피드백
                            if landmarks[26].x > landmarks[32].x:  # 앞무릎이 앞발보다 앞에 있을때
                                results_dict["무릎이 너무 앞으로 나갔어요. 조금만 뒤로"] += 1
                                feedback_messages.append("무릎이 너무 앞으로 나갔어요. 조금만 뒤로")

                            # 앞무릎 각도 피드백
                            if right_knee_angle > 95:  # 앞무릎이 90도 보다 클때
                                results_dict["앞무릎을 90도로 만들어 주세요."] += 1
                                feedback_messages.append("앞무릎을 90도로 만들어 주세요.")

                            # 뒷무릎 피드백
                            if left_knee_angle < 95:  # 뒷무릎이 90도에 가까울때
                                results_dict["뒷무릎 너무 내려가지 마세요."] += 1
                                feedback_messages.append("뒷무릎 너무 내려가지 마세요.")

                            # 상체 피드백
                            # if abs(landmark[0] - waist_x) > 0.1:  # 상체가 똑바로 서 있지 않을 때
                            if abs(torso_angle) > 0.1:  # 상체가 똑바로 서 있지 않을 때 
                                results_dict["허리 쭉 펴고 상체를 바르게 해주세요."] += 1
                                feedback_messages.append("허리 쭉 펴고 상체를 바르게 해주세요.")
                            
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
                        # draw.text((x1, y1 - 40), f"Count: {side_lunge_count} | Correct Ratio: {correct_ratio:.2f} | {prediction_text}", font=font, fill=(255, 0, 0))
                        # frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                        # mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # 횟수 표시
                        count_placeholder.info(f"런지 횟수 : {lunge_count}/{target_reps}     런지 오자세 횟수 : {wrong_count}    세트 : {completed_sets}/{target_sets}")
                        # count_placeholder.info(f"런지 횟수 : {lunge_count}/{target_reps}   세트 : {completed_sets}/{target_sets}")

                    except IndexError:
                        print("IndexError: 랜드마크 데이터 추출 오류")

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    results_dict["정자세 횟수"] = completed_sets * target_reps
    results_dict['오자세 횟수'] = wrong_count
    
    total_feedback = results_dict['무릎이 너무 앞으로 나갔어요. 조금만 뒤로'] + results_dict['앞무릎을 90도로 만들어 주세요.'] + results_dict['뒷무릎 너무 내려가지 마세요.'] + results_dict['허리 쭉 펴고 상체를 바르게 해주세요.']
    results_dict['무릎이 너무 앞으로 나갔어요. 조금만 뒤로'] = round(results_dict['무릎이 너무 앞으로 나갔어요. 조금만 뒤로'] / total_feedback, 5)
    results_dict['앞무릎을 90도로 만들어 주세요.'] = round(results_dict['앞무릎을 90도로 만들어 주세요.'] / total_feedback , 5)
    results_dict['뒷무릎 너무 내려가지 마세요.'] = round(results_dict['뒷무릎 너무 내려가지 마세요.'] / total_feedback, 5)
    results_dict['허리 쭉 펴고 상체를 바르게 해주세요.'] = round(results_dict['허리 쭉 펴고 상체를 바르게 해주세요.'] / total_feedback, 5)
    

    return results_dict