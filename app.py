import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import pandas as pd
import math

st.set_page_config(page_title="المختبر البايوميكانيكي", layout="wide")
st.title("🔬 المختبر البايوميكانيكي للوثب العالي")

# --- حماية ذاكرة السيرفر (Caching) ---
@st.cache_resource
def load_mediapipe_model():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing

mp_pose, pose, mp_drawing = load_mediapipe_model()

# --- إعدادات المعايرة ---
st.markdown("### 📋 إعدادات المعايرة (Calibration)")
col1, col2, col3 = st.columns(3)
player_height_cm = col1.number_input("طول اللاعب (سم):", min_value=140.0, max_value=230.0, value=185.0)
player_weight_kg = col2.number_input("وزن اللاعب (كجم):", min_value=40.0, max_value=120.0, value=75.0)
video_fps = col3.number_input("سرعة إطارات الفيديو (FPS):", min_value=10, max_value=240, value=30)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

uploaded_file = st.file_uploader("🎥 ارفع فيديو القفزة هنا", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st.markdown("---")
    vid_col, data_col = st.columns([2, 1])
    st_frame = vid_col.empty()
    live_data = data_col.empty()
    
    research_data = []
    pixel_to_meter, prev_com_x, prev_com_y = None, None, None
    frame_count = 0
    time_step = 1.0 / video_fps
    
    st.info("🔄 جاري تحليل الفيديو... يرجى الانتظار (قد يستغرق وقتاً أطول قليلاً على السحابة)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_time = frame_count * time_step
        
        # تصغير الفيديو لتقليل الضغط على السيرفر
        height, width, _ = frame.shape
        frame = cv2.resize(frame, (640, int(640 * (height/width))))
        h, w, _ = frame.shape
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w, landmarks[mp_pose.PoseLandmark.NOSE.value].y * h]
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
            
            if pixel_to_meter is None and frame_count < 10:
                player_pixel_height = abs(ankle[1] - nose[1])
                if player_pixel_height > 0:
                    pixel_to_meter = (player_height_cm / 100.0) / player_pixel_height
            
            if pixel_to_meter is not None:
                com_x, com_y = hip[0], hip[1]
                com_height_meters = (h - com_y) * pixel_to_meter
                
                vel_x, vel_y = 0.0, 0.0
                if prev_com_x is not None:
                    vel_x = ((com_x - prev_com_x) * pixel_to_meter) / time_step
                    vel_y = ((prev_com_y - com_y) * pixel_to_meter) / time_step
                
                prev_com_x, prev_com_y = com_x, com_y
                total_velocity = math.sqrt(vel_x**2 + vel_y**2)
                momentum = player_weight_kg * total_velocity
                
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)
                trunk_lean = calculate_angle(shoulder, hip, [hip[0], hip[1] - 100])
                
                research_data.append({
                    "رقم الإطار": frame_count, "الزمن (ث)": round(current_time, 3),
                    "زاوية الركبة": round(knee_angle, 2), "زاوية الحوض": round(hip_angle, 2),
                    "ميل الجذع": round(trunk_lean, 2), "ارتفاع الثقل (م)": round(com_height_meters, 2),
                    "السرعة (م/ث)": round(total_velocity, 2), "كمية الحركة": round(momentum, 2)
                })
                
                with live_data.container():
                    st.markdown("### 📊 القراءات الحالية")
                    st.write(f"⏱️ **الزمن:** {round(current_time, 2)} ث")
                    st.write(f"📐 **الركبة:** {int(knee_angle)}° | **الحوض:** {int(hip_angle)}°")
                    st.write(f"🚀 **السرعة:** {round(total_velocity, 2)} م/ث")
                    st.write(f"📏 **الارتفاع:** {round(com_height_meters, 2)} م")
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
        st_frame.image(image, channels="BGR", use_column_width=True)

    cap.release()
    st.success("✅ اكتمل التحليل!")
    
    if len(research_data) > 0:
        df = pd.DataFrame(research_data)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 تحميل النتائج (Excel)", data=csv, file_name='HighJump_Data.csv', mime='text/csv')
