import cv2
import face_recognition

print("Libraries ellam pakkava import aagiduchu macha!")

# Webcam on panrom (0 = Default Laptop Camera)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Camera open aagala!")
        break
        
    # Screen-la un video-va kaata
    cv2.imshow("Camera Test - Press 'q' to exit", frame)
    
    # Keyboard-la 'q' amukkuna veliya vandhudum
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()