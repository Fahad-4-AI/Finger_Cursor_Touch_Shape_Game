import cv2
import mediapipe as mp
import pyautogui
import random
import numpy as np
import math

# Set up the camera and hand tracking
cam = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

shapes = []  # List to store information about the falling shapes

def generate_shape():
    shape_type = random.choice(["circle", "rectangle", "triangle"])
    color = None
    if shape_type == "circle":
        color = (0, 255, 0)  # Green
    elif shape_type == "rectangle":
        color = (0, 0, 255)  # Red
    elif shape_type == "triangle":
        color = (0, 255, 255)  # Yellow

    size = random.randint(20, 50)  # Random size
    x = random.randint(0, screen_width)
    y = 0  # Start from the top of the screen
    speed = random.randint(2, 5)  # Random falling speed

    return {"type": shape_type, "color": color, "size": size, "x": x, "y": y, "speed": speed}

index_x, index_y = screen_width // 2, screen_height // 2  # Initialize mouse position

score = 0  # Initialize the score variable
clicked = False  # Initialize the click status

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            thumb = landmarks[4]  # Thumb landmark
            index = landmarks[8]  # Index finger landmark

            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)
            index_x = int(index.x * frame_width)
            index_y = int(index.y * frame_height)

            # Calculate the distance between thumb and index finger
            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            if distance < 30:  # Adjust the distance threshold as needed
                # Perform a click action here
                if not clicked:
                    pyautogui.click()
                    clicked = True
            else:
                clicked = False

            # Move the mouse cursor according to the index finger position
            pyautogui.moveTo(index_x, index_y)

    # Check if the cursor is directly over a shape and update the score
    for shape in shapes:
        shape_x, shape_y, shape_size = shape["x"], shape["y"], shape["size"]
        if shape_x <= index_x <= shape_x + shape_size and shape_y <= index_y <= shape_y + shape_size:
            if shape["type"] == "circle":
                score += 5
            elif shape["type"] == "rectangle":
                score -= 3
            elif shape["type"] == "triangle":
                score -= 2
            shapes.remove(shape)

    # Generate new shapes with a higher probability
    if random.random() < 0.05:  # Adjust the probability as needed
        shapes.append(generate_shape())

    # Update and draw shapes
    for shape in shapes:
        shape["y"] += shape["speed"]
        if shape["type"] == "circle":
            cv2.circle(frame, (int(shape["x"]), int(shape["y"])), shape["size"], shape["color"], -1)
        elif shape["type"] == "rectangle":
            cv2.rectangle(frame, (int(shape["x"]), int(shape["y"])), (int(shape["x"] + shape["size"]), int(shape["y"] + shape["size"])), shape["color"], -1)
        elif shape["type"] == "triangle":
            points = np.array([(int(shape["x"]), int(shape["y"])), (int(shape["x"] + shape["size"]), int(shape["y"])), (int(shape["x"] + shape["size"] / 2), int(shape["y"] + shape["size"]))])
            cv2.drawContours(frame, [points], 0, shape["color"], -1)

    # Remove shapes that have gone off the screen
    shapes = [shape for shape in shapes if shape["y"] < screen_height]

    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Check if the player has won
    if score >= 20:
        cv2.putText(frame, "You Win!", (frame_width // 2 - 100, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow('Finger Mouse', frame)
    cv2.waitKey(1)
