import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime

# 1. Webcam on panrom
video_capture = cv2.VideoCapture(0)

# 2. Photos folder-la irundhu images load panrom & encode panrom
print("Photos load aagudhu macha, konjam wait pannu...")

thameem_image = face_recognition.load_image_file("photos/thameem.jpg")
thameem_encoding = face_recognition.face_encodings(thameem_image)[0]

dhanush_image = face_recognition.load_image_file("photos/dhanush.jpg")
dhanush_encoding = face_recognition.face_encodings(dhanush_image)[0]

dq_image = face_recognition.load_image_file("photos/dq.jpg")
dq_encoding = face_recognition.face_encodings(dq_image)[0]

surya_image = face_recognition.load_image_file("photos/surya.jpg")
surya_encoding = face_recognition.face_encodings(surya_image)[0]

# Encodings & Names-ah list-la podrom
known_face_encodings = [
    thameem_encoding,
    dhanush_encoding,
    dq_encoding,
    surya_encoding
]

known_face_names = [
    "Thameem",
    "Dhanush",
    "Dulquer",
    "Surya"
]

# Attendance podradhukkaga copy list (Oru thadava attendance vizhundha, idhula irundhu remove panniduvom)
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

# 3. Innikku Date eduthu CSV file create panrom
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)
# Excel sheet-la mela Heading varadhukku
lnwriter.writerow(["Name", "Time"])

print("System Ready! Camera open aagudhu...")

# 4. Main Webcam Loop
while True:
    _, frame = video_capture.read()
    
    # Fast processing-kaga frame size-ah 1/4th aa kuraikkurom
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # OpenCV BGR format-la irundhu RGB format-ku mathurom
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
            
            # Mugam match aana CSV-la entry podrom
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
                    f.flush() # Udane CSV file-la save aaga
                    print(f"✅ Attendance Marked: {name} at {current_time}")

    # Mugathai suthi Green Box & Name kaata
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow("Attendance System - Press 'q' to Quit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()