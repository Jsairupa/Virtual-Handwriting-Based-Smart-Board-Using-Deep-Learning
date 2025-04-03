
###  IMPORT THE NECESSARY LIBRARIES
import cv2
from own_hand_detector import HandDetector
import numpy as np

### INITIALIZE THE BOARD ALONG WITH THE DIMENSIONS
board = np.zeros((480,640,3),np.uint8)

### INITIALIZE THE HAND DETECTOR 
detector = HandDetector(maxHands=1, detectionCon=0.8)

### ESTABLISH THE CONNECTION BETWEEN THE CAMERA HARDWARE AND THE SOFTWARE LIBRARY
video = cv2.VideoCapture(0)


### RESIZE THE CAMERA TO THE GIVEN BELOW DIMENSIONS
video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


### ALGORITHM FLOW OF THE PROPOSED VIRTUAL TEXT WRITING CODE
state = "working"
while True:
    _, img = video.read()
    img = cv2.flip(img, 1)
    hand = detector.findHands(img, draw=False)
    if hand:
        lmlist = hand[0]
        coords = lmlist['lmList']
        if lmlist:
            fingerup = detector.fingersUp(lmlist)
            if fingerup == [0, 1, 0, 0, 0]:
                state = "write"
                board = cv2.circle(board, (coords[8][0],coords[8][1]),5, (250,160,90),-1)
            if fingerup == [0, 1, 1, 0, 0]:
                state = "erase"
                board = cv2.circle(board, (coords[8][0],coords[8][1]),5, (250,160,90),-1)
            if fingerup == [1, 1, 0, 0, 0]:
                state = "move"
                prev_board = board
                cv2.circle(board, (coords[8][0],coords[8][1]),5, (250,160,90),-1)
            if fingerup == [1, 1, 1, 1, 1]:
                state = "clean"
                board[:,:,:] = 0
            print(coords[8])
    
    cv2.imshow("Video", img)
    cv2.imshow("board",board)
    
    
    #### WRITING FUNCTIONALITY BASED ON THE STATE GIVEN BY THE FINGERTIPS
    if state == "write":
        board = cv2.circle(board,(coords[8][0],coords[8][1]),5,(255,255,255),-1)
    if state == "erase":
        board = cv2.circle(board,(coords[8][0],coords[8][1]),5,(0,0,0),-1)
    if state == "move":
        board = prev_board
        #cv2.circle(board,(coords[8][0],coords[8][1]),5,(0,0,0),-1)
        #if values!=0:
        #    board = cv2.circle(board,(coords[8][0],coords[8][1]),5,(255,255,255),-1)
        #    print("YES")
        #else:
        #    board = cv2.circle(board,(coords[8][0],coords[8][1]),5,(0,0,0),-1)
    
    
    
    ### EXIT CONDITION FOR BREAKING THE LOOP
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(state)
    
### TERMINATE THE CONNECTION BETWEEN THE CAMERA AND THE SOFTWARE LIBRARIES
video.release()
cv2.destroyAllWindows()