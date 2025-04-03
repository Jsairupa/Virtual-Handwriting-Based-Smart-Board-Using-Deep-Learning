import cv2
import numpy as np
from hand_detector.detector import YOLO
from unified_detector import Fingertips

hand_detector_model = YOLO(weights = 'weights/yolo.h5', threshold = 0.8)

fingertips = Fingertips(weights='weights/fingertip.h5')

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    
    if ret is False:
        break
    
    left_coordinate, right_coordinate = hand_detector_model.detect(img)
    
    if left_coordinate and right_coordinate is not None:
        cropped_image = img[left_coordinate[1]:right_coordinate[1], left_coordinate[0]: right_coordinate[0]]
        height, width, _ = cropped_image.shape
    
        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)
    
        # post-processing
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + left_coordinate[0]
            pos[i + 1] = pos[i + 1] * height + left_coordinate[1]
    
        # drawing
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        img = cv2.rectangle(img, (left_coordinate[0], left_coordinate[1]), (left_coordinate[0], right_coordinate[1]), (235, 26, 158), 2)
        for c, p in enumerate(prob):
            if p > 0.5:
                img = cv2.circle(img, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
            index = index + 2

    cv2.imshow("image", img)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
 
camera.release()
cv2.destroyAllWindows()
