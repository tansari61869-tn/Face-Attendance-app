import streamlit as st
import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime
from PIL import Image
import os

st.set_page_config(page_title="Face Attendance System", page_icon="📸", layout="centered")

st.title("📸 Smart Facial Recognition Attendance")
st.write("Camera-la unga mugathai kaati 'Take Photo' click pannunga, automatic-aa attendance vizhundhudum!")

# 1. Bulletproof Image Loader (Always forces 8-bit RGB for dlib)
def load_clean_rgb(path):
    pil_img = Image.open(path).convert("RGB")
    return np.array(pil_img, dtype=np.uint8)

@st.cache_resource
def load_known_faces():
    known_encodings = []
    known_names = []
    
    people = [
        ("Thameem", "photos/thameem.jpg"),
        ("Dhanush", "photos/dhanush.jpg"),
        ("Dulquer", "photos/dq.jpg"),
        ("Surya", "photos/surya.jpg")
    ]
    
    for name, path in people:
        if os.path.exists(path):
            img = load_clean_rgb(path)
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0:
                known_encodings.append(enc[0])
                known_names.append(name)
                
    return known_encodings, known_names

known_face_encodings, known_face_names = load_known_faces()

# 2. CSV File Setup (Pure Python CSV - No Pandas needed!)
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date"])

def get_marked_names():
    names = []
    if os.path.exists(csv_file):
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # Header skip panrom
            for row in reader:
                if len(row) > 0:
                    names.append(row[0])
    return names

# 3. Browser Webcam Input
img_file_buffer = st.camera_input("Look at the camera and click Take Photo")

if img_file_buffer is not None:
    pil_image = Image.open(img_file_buffer).convert("RGB")
    rgb_img = np.array(pil_image, dtype=np.uint8)
    cv2_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    
    if len(face_encodings) == 0:
        st.warning("⚠️ Moonji theliva theriyala! Konjam velichathula ninnu thirumba photo edu.")
    else:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            # Photo-la Box & Name varairom
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(cv2_img, (left, top), (right, bottom), color, 2)
            cv2.rectangle(cv2_img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(cv2_img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            if name != "Unknown":
                marked_names = get_marked_names()
                if name not in marked_names:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, current_time, current_date])
                    st.success(f"✅ Attendance Marked for **{name}** at {current_time}!")
                else:
                    st.info(f"ℹ️ **{name}**, Thambi marakkama daily attendance podanum innaikku nee present po...!")
            else:
                st.error("❌ Unknown Face! Database-la indha moonji ah illa.")
                
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Processed Frame", use_container_width=True)

# 4. Live Attendance Sheet Display & Download Button
st.divider()
st.subheader(f"📋 Today's Attendance Sheet ({current_date})")

if os.path.exists(csv_file):
    with open(csv_file, "r", newline="") as f:
        reader = list(csv.reader(f))
        if len(reader) > 1:
            headers = reader[0]
            rows = reader[1:]
            
            # Pandas thevai illama Markdown vechu table varairom!
            md_table = "| " + " | ".join(headers) + " |\n"
            md_table += "|" + "|".join(["---"] * len(headers)) + "|\n"
            for row in rows:
                if len(row) > 0:
                    md_table += "| " + " | ".join(row) + " |\n"
            
            st.markdown(md_table)
        else:
            st.write("Innum yaarukkum attendance vizhala. Photo edunga!")
            
    with open(csv_file, "rb") as file:
        st.download_button(
            label="📥 Download Attendance CSV",
            data=file,
            file_name=csv_file,
            mime="text/csv"
        )