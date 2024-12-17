# pushup.py
import cv2
import joblib
import math
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import mediapipe as mp
import time
from ultralytics import YOLO
from collections import namedtuple

label = ['정자세', '오자세']
model = YOLO('yolov5/yolo11n.pt')
pushup_model = joblib.load('C:/ai5/본프로젝트/메인/_data/운동별 모델/squat.pkl')
font_path = "C:/ai5/본프로젝트/메인/_data/강원교육튼튼.ttf"
font = ImageFont.truetype(font_path, 20)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def squat_info(st):
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
                        <h2 style="font-size: 20px; color: #333; margin: 0; text-align: center;">스쿼트</h3>
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
        st.image("C:/ai5/본프로젝트/메인/_data/스쿼트.png", use_container_width=True)
    
        if st.button("다음"):
            st.session_state.current_exercise_state = "camera"
            st.rerun()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)


def calculate_angle(point_a, point_b, point_c):
    def get_coords(point):
        if hasattr(point, 'x'):  # Mediapipe 객체
            return point.x, point.y, point.z
        elif isinstance(point, (list, tuple)):  # 리스트나 튜플
            return point[0], point[1], point[2]
        else:
            raise ValueError("Unsupported point type")

    ax, ay, az = get_coords(point_a)
    bx, by, bz = get_coords(point_b)
    cx, cy, cz = get_coords(point_c)

    ab = (bx - ax, by - ay, bz - az)
    bc = (cx - bx, cy - bz, cz - bz)
    dot_product = sum(a * b for a, b in zip(ab, bc))
    magnitude_ab = math.sqrt(sum(a**2 for a in ab))
    magnitude_bc = math.sqrt(sum(b**2 for b in bc))

    if magnitude_ab == 0 or magnitude_bc == 0:
        print(f"Warning: Zero vector detected. Points: A={point_a}, B={point_b}, C={point_c}")
        return 0.0  # 각도 계산 불가능

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

def run_squat_session(st, YOLO, info):
    # 초기 변수 설정
    total_history = []  # 전체 프레임의 예측 결과 저장
    correct_threshold = 0.5  # 정자세 비율 기준 (70%)
    squat_count = 0
    squat_wrong_count = 0
    current_state = "업"
    
    # 세트 및 횟수 설정
    target_reps = info["횟수"]  # 목표 횟수
    target_sets = info["세트"]  # 목표 세트
    completed_sets = 0  # 완료된 세트 수

    frame_skip = 4  # 1 프레임 건너뛰기
    frame_count = 0  # 프레임 카운터 초기화

    #### 결과 저장용 딕셔너리 초기화
    results_dict = {
        "정자세 횟수": 0,
        "오자세 횟수": 0,
        "무릎이 발끝을 넘지 않도록 하세요.": 0,
        "허리를 더 펴세요.": 0,
        "상체를 더 세우세요.": 0,
        "무릎, 엉덩이, 발을 일직선으로 정렬하세요.": 0
    }

    # MediaPipe Pose 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # 스트림릿 화면 초기화
    st.markdown("<h1 style='text-align: center;'>스쿼트</h1>", unsafe_allow_html=True)

    # 스트리밍 비디오 피드 설정
    video_url = "C:/ai5/본프로젝트/메인/영상/스쿼트/사영2.mp4"
    # video_url = "C:/ai5/본프로젝트/메인/_data/스쿼트/정자세/스쿼트4.mp4"
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
                        
                        # 좌우 반전 감지
                        nose_x = landmarks[0].x
                        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                        is_left_facing = nose_x < hip_mid_x

                        if is_left_facing:
                            landmarks = flip_landmarks(landmarks)

                        # 랜드마크 데이터 생성
                        landmarks_data = []
                        for idx in [11, 12, 23, 24, 25, 26, 27, 28]:
                            landmarks_data.extend([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

                        # 목 좌표 계산
                        neck_x = (landmarks[11].x + landmarks[12].x) / 2
                        neck_y = (landmarks[11].y + landmarks[12].y) / 2
                        neck_z = (landmarks[11].z + landmarks[12].z) / 2
                        neck = [neck_x, neck_y, neck_z]
                        landmarks_data.extend([neck_x, neck_y, neck_z])

                        # 허리 좌표 계산
                        waist_x = (landmarks[23].x + landmarks[24].x) / 2
                        waist_y = (landmarks[23].y + landmarks[24].y) / 2
                        waist_z = (landmarks[23].z + landmarks[24].z) / 2
                        Landmark = namedtuple("Landmark", ["x", "y", "z"])
                        waist = Landmark(x=waist_x, y=waist_y, z=waist_z)
                        landmarks_data.extend([waist_x, waist_y, waist_z])

                        # 각도 계산
                        left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                        right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                        left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                        right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])

                        pelvis_balance = abs(landmarks[23].y - landmarks[24].y)

                        # 상체 관련 추가 피처
                        waist_to_hip_angle = calculate_angle(waist, [waist[0], waist[1] - 1, waist[2]], landmarks[23])
                        shoulder_tilt = abs(landmarks[11].y - landmarks[12].y)
                        hip_tilt = abs(landmarks[23].y - landmarks[24].y)
                        upper_body_balance = shoulder_tilt - hip_tilt
                        
                            # 데이터 추가
                        landmarks_data.extend([
                            left_knee_angle, right_knee_angle,
                            left_hip_angle, right_hip_angle,
                            pelvis_balance, waist_to_hip_angle, shoulder_tilt, hip_tilt, upper_body_balance
                        ])
                                        
                        # 상체 관련 추가 피처
                        pelvis_center = Landmark(
                            x=(landmarks[23].x + landmarks[24].x) / 2,
                            y=(landmarks[23].y + landmarks[24].y) / 2,
                            z=(landmarks[23].z + landmarks[24].z) / 2
                        )
                        head_to_waist_angle = calculate_angle(landmarks[0], waist, pelvis_center)
                        head_waist_y_difference = abs(landmarks[0].y - waist.y)
                        neck_to_waist_angle = calculate_angle(neck, waist, pelvis_center)
                        left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
                        right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
                        left_back_incline_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[27])
                        right_back_incline_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[28])
                        left_mid_spine = Landmark(
                            x=(landmarks[11].x + landmarks[23].x) / 2,
                            y=(landmarks[11].y + landmarks[23].y) / 2,
                            z=(landmarks[11].z + landmarks[23].z) / 2
                        )
                        right_mid_spine = Landmark(
                            x=(landmarks[12].x + landmarks[24].x) / 2,
                            y=(landmarks[12].y + landmarks[24].y) / 2,
                            z=(landmarks[12].z + landmarks[24].z) / 2
                        )
                        left_spine_distance = calculate_distance(landmarks[23], left_mid_spine)
                        right_spine_distance = calculate_distance(landmarks[24], right_mid_spine)
                        # 어깨와 엉덩이 중심 계산
                        shoulder_mid = Landmark(
                            x = (landmarks[11].x + landmarks[12].x) / 2, 
                            y = (landmarks[11].y + landmarks[12].y) / 2,
                            z = (landmarks[11].z + landmarks[12].z) / 2)
                    
                        hip_mid = Landmark(
                            x = (landmarks[23].x + landmarks[24].x) / 2, 
                            y = (landmarks[23].y + landmarks[24].y) / 2,
                            z = (landmarks[23].z + landmarks[24].z) / 2)
                        
                        back_mid = Landmark(
                            x = (landmarks[11].x + landmarks[23].x) / 2, 
                            y = (landmarks[11].y + landmarks[23].y) / 2,
                            z = (landmarks[11].z + landmarks[23].z) / 2)
                        
                        # 허리와 어깨의 벡터 차이
                        shoulder_vector = (landmarks[11].x - landmarks[12].x, landmarks[11].y - landmarks[12].y, landmarks[11].z - landmarks[12].z)
                        hip_vector = (landmarks[23].x - landmarks[24].x, landmarks[23].y - landmarks[24].y, landmarks[24].z - landmarks[24].z)
                        shoulder_to_hip_angle = calculate_angle(shoulder_mid, hip_mid, back_mid)

                        # 벡터 차이 계산
                        vector_diff = abs(shoulder_vector[0] - hip_vector[0]) + abs(shoulder_vector[1] - hip_vector[1])        
                        # 디버깅 출력
                        print(f"Head to Waist Angle: {head_to_waist_angle}")
                        print(f"Head to Waist Y Difference: {head_waist_y_difference}")
                        print(f"Neck to Waist Angle: {neck_to_waist_angle}")

                        # 데이터 추가
                        landmarks_data.extend([head_to_waist_angle, head_waist_y_difference, neck_to_waist_angle, left_hip_angle, right_hip_angle,
                                            left_back_incline_angle, right_back_incline_angle, left_spine_distance, right_spine_distance, shoulder_to_hip_angle,
                                                vector_diff ])
                        # 자세 예측
                        prediction = pushup_model.predict([landmarks_data])
                        prediction_text = label[int(prediction[0])]

                        # 전체 기록에 추가
                        total_history.append(prediction_text == "정자세")

                        # 정자세 비율 계산
                        correct_ratio = sum(total_history) / len(total_history)
                        print(f"Correct Ratio: {correct_ratio:.2f}")

                        if right_knee_angle > 85:  # 무릎 각도가 20도 미만일 때 "업" 상태
                            if current_state == "업":  # 이전 상태가 "업"이 아니었다면 전환
                                current_state = "다운"

                        elif right_knee_angle <= 85:  # 무릎 각도가 40도 이상일 때 "다운" 상태
                            if current_state == "다운":
                                if correct_ratio >= correct_threshold:
                                    squat_count += 1
                                    current_state = "업"

                                    if squat_count == target_reps:
                                        completed_sets += 1
                                        squat_count = 0

                                        if completed_sets == target_sets:
                                            st.success("모든 세트를 완료했습니다! 수고하셨습니다!")
                                            cap.release()
                                            cv2.destroyAllWindows()
                                            results_dict["정자세 횟수"] = completed_sets * target_reps
                                            return results_dict
                                else:
                                    squat_wrong_count += 1
                                total_history = []  # 새로운 푸시업을 기준으로 기록 초기화

                        # "오자세"인 경우 피드백 생성 및 출력
                        if prediction_text == "오자세" :
                            feedback_messages = []
                            
                            # 무릎-발 간 거리 체크
                            if landmarks[25].x > landmarks[31].x:  # 무릎이 발끝을 넘으면
                                results_dict["무릎이 발끝을 넘지 않도록 하세요."] += 1
                                feedback_messages.append("무릎이 발끝을 넘지 않도록 하세요.")

                            # 허리-엉덩이 각도 체크
                            waist_to_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])  # 왼쪽 허리-엉덩이-무릎
                            if waist_to_hip_angle < 90:  # 허리가 과도하게 굽혀진 경우
                                results_dict["허리를 더 펴세요."] += 1
                                feedback_messages.append("허리를 더 펴세요.")

                            # 상체 기울기 체크
                            upper_body_angle = calculate_angle(landmarks[0], landmarks[11], landmarks[23])  # 목-허리-엉덩이
                            if upper_body_angle < 160:  # 상체가 너무 기울어진 경우
                                results_dict["상체를 더 세우세요."] += 1
                                feedback_messages.append("상체를 더 세우세요.")

                            # 무릎-엉덩이-발 정렬 체크
                            knee_hip_foot_alignment_angle = calculate_angle(landmarks[25], landmarks[23], landmarks[31])  # 왼쪽 무릎-엉덩이-발
                            if abs(knee_hip_foot_alignment_angle - 180) > 10:  # 일직선이 아니면
                                results_dict["무릎, 엉덩이, 발을 일직선으로 정렬하세요."] += 1
                                feedback_messages.append("무릎, 엉덩이, 발을 일직선으로 정렬하세요.")

                            
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
                        count_placeholder.info(f"스쿼트 횟수 : {squat_count}/{target_reps}     스쿼트 오자세 횟수 : {squat_wrong_count}      세트 : {completed_sets}/{target_sets}")
                        # count_placeholder.info(f"스쿼트 횟수 : {squat_count}/{target_reps}    세트 : {completed_sets}/{target_sets}")

                    except IndexError:
                        print("IndexError: 랜드마크 데이터 추출 오류")

        # 회전된 프레임을 표시
        frame_placeholder.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    results_dict["정자세 횟수"] = completed_sets * target_reps
    results_dict['오자세 횟수'] = squat_wrong_count
    
    total_feedback = results_dict['무릎이 발끝을 넘지 않도록 하세요.'] + results_dict['허리를 더 펴세요.'] + results_dict['상체를 더 세우세요.'] + results_dict['무릎, 엉덩이, 발을 일직선으로 정렬하세요.']
    results_dict['무릎이 발끝을 넘지 않도록 하세요.'] = round(results_dict['무릎이 발끝을 넘지 않도록 하세요.'] / total_feedback, 5)
    results_dict['허리를 더 펴세요.'] = round(results_dict['허리를 더 펴세요.'] / total_feedback , 5)
    results_dict['상체를 더 세우세요.'] = round(results_dict['상체를 더 세우세요.'] / total_feedback, 5)
    results_dict['무릎, 엉덩이, 발을 일직선으로 정렬하세요.'] = round(results_dict['무릎, 엉덩이, 발을 일직선으로 정렬하세요.'] / total_feedback, 5)
    

    return results_dict
