import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Check if the camera is available
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Failed to open camera. Please ensure your camera is properly connected.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'E', 4: 'C', 5: 'H', 6: 'I', 7: 'V', 8: 'U', 9: 'W', 10: 'F', 11: 'S', 12: 'R', 13: 'D', 14: 'O'}  # Update with 'U'

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame. Skipping...")
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx > 0:
                # Skip processing second hand
                continue

            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Change color to green
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Check if the number of features matches the model's input
            if len(data_aux) != 42:
                print("Warning: Hand has a different number of features. Skipping...")
                continue

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  # Change color to green

    # Resize the frame to match screen resolution
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    cv2.imshow('frame', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
