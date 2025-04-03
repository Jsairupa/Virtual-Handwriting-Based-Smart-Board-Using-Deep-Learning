import cv2
from own_hand_detector import HandDetector
import numpy as np
import datetime

# Create canvas
board = np.zeros((480, 640, 3), np.uint8)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Setup webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

state = "working"
values = np.array([0, 0, 0])
prev_point = None
frame_count = 0
drawing = False
fail_count = 0

while True:
    success, img = video.read()

    # Reinitialize camera if it fails
    if not success:
        fail_count += 1
        print(f"⚠️ Frame grab failed ({fail_count}). Retrying...")
        if fail_count >= 10:
            print("🔄 Restarting camera...")
            video.release()
            video = cv2.VideoCapture(0)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            fail_count = 0
        continue
    else:
        fail_count = 0

    img = cv2.flip(img, 1)
    frame_count += 1

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        coords = hand['lmList']

        if coords and len(coords) > 8:
            x, y = coords[8][0], coords[8][1]
            x = min(max(x, 0), 639)
            y = min(max(y, 0), 479)

            fingerup = detector.fingersUp(hand)

            if frame_count % 10 == 0:
                print("🖐️ Fingers:", fingerup, "| ✏️ State:", state, "| 👉", (x, y))

            if fingerup == [0, 1, 0, 0, 0]:  # Write
                state = "write"
                drawing = True
            elif fingerup == [0, 1, 1, 0, 0]:  # Erase
                state = "erase"
                board = cv2.circle(board, (x, y), 20, (0, 0, 0), -1)
                drawing = False
                prev_point = None
            elif fingerup == [1, 1, 0, 0, 0]:  # Move
                state = "move"
                values = board[y, x, 0]
                drawing = False
                prev_point = None
            elif fingerup == [1, 1, 1, 1, 1]:  # Clear board
                state = "clean"
                board[:, :, :] = 0
                drawing = False
                prev_point = None
            else:
                drawing = False
                prev_point = None

            if drawing:
                if prev_point:
                    board = cv2.line(board, prev_point, (x, y), (255, 255, 255), 5)
                prev_point = (x, y)
        else:
            drawing = False
            prev_point = None
    else:
        drawing = False
        prev_point = None

    # Show windows
    cv2.imshow("Video", img)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting...")
        break

    if key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"my_drawing_{timestamp}.png"
        success = cv2.imwrite(filename, board)
        if success:
            print(f"✅ Drawing saved as: {filename}")
        else:
            print("❌ Failed to save drawing!")

    if key == ord('c'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined = np.hstack((img, board))
        filename = f"combined_{timestamp}.png"
        cv2.imwrite(filename, combined)
        print(f"Combined screenshot saved as {filename}")

# Cleanup
video.release()
cv2.destroyAllWindows()
