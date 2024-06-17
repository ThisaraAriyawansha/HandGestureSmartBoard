import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

drawing = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None
clean_board = False

def detect_hand(frame):
    global prev_x, prev_y, drawing, clean_board

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the original frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the finger tips
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            h, w, _ = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Check if index and middle fingers are up, and thumb, ring, and pinky fingers are down
            if (index_finger_tip.y < 0.5 and middle_finger_tip.y < 0.5 and
                thumb_tip.y > 0.6 and ring_finger_tip.y > 0.6 and pinky_tip.y > 0.6):
                # Clear the drawing if the gesture is detected
                drawing = np.zeros((480, 640, 3), dtype=np.uint8)
                prev_x, prev_y = None, None
                clean_board = True
            else:
                clean_board = False
                # Draw a line from the previous point to the current point if only one finger is raised
                if prev_x is not None and prev_y is not None:
                    cv2.line(drawing, (prev_x, prev_y), (index_x, index_y), (255, 255, 255), 4)

                prev_x, prev_y = index_x, index_y

    # Combine the original frame with the drawing
    combined_frame = np.hstack((frame, drawing))

    return combined_frame

def gen_frames():
    global clean_board
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            drawing_frame = detect_hand(frame)
            if clean_board:
                # Show number 2 on the board
                cv2.putText(drawing_frame, '2', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', drawing_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
