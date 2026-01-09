import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

cap_1 = cv2.VideoCapture(2) # Use 1 for angle camera 
cap_2 = cv2.VideoCapture(1) # Use 2 for visuotactile camera

cap_1.set(cv2.CAP_PROP_EXPOSURE, -7)

# Make Detections from Webcam Input
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.8) as pose:
    while cap_1.isOpened():
        ret1, frame1 = cap_1.read()
        ret2, frame2 = cap_2.read()

        if not ret1 or not ret2:
            print("Failed to grab frames")
            break

        # Recolor image to RGB - pass to mediapipe
        image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection -> store detections in an array named 'results'
        results = pose.process(image)
        
        # Recolor back to BGR for rendering for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections for whole:
        #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ''' Extract landmarks (OG code for angle calculation)
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for right leg
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Visualize angle
            cv2.putText(image, str(right_knee_angle), 
                        tuple(np.multiply(right_knee, [image.shape[1], image.shape[0]]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass '''
        
        # --- Extract landmarks and draw only hip–knee–ankle with angle ---
        try:
            landmarks = results.pose_landmarks.landmark

            # Normalized coordinates (0–1)
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]
            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]

            # Convert to pixel coordinates
            h, w, _ = image.shape
            hip_px   = tuple((np.array(right_hip)   * [w, h]).astype(int))
            knee_px  = tuple((np.array(right_knee)  * [w, h]).astype(int))
            ankle_px = tuple((np.array(right_ankle) * [w, h]).astype(int))

            # Calculate angle
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

            # --- Draw ONLY hip–knee–ankle ---

            # Circles at joints
            cv2.circle(image, hip_px,   6, (0, 255, 0), -1)
            cv2.circle(image, knee_px,  6, (0, 255, 0), -1)
            cv2.circle(image, ankle_px, 6, (0, 255, 0), -1)

            # Lines between joints
            cv2.line(image, hip_px,  knee_px,  (255, 0, 0), 3)
            cv2.line(image, knee_px, ankle_px, (255, 0, 0), 3)

            # Visualize angle near knee
            cv2.putText(
                image, f"{right_knee_angle:.1f}",
                (knee_px[0] + 10, knee_px[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        except Exception as e:
            # You can print(e) for debugging if needed
            pass

        cv2.imshow('Webcam Feed', image)
        cv2.imshow('Visuotactile Camera', frame2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap_1.release()
    cv2.destroyAllWindows()



