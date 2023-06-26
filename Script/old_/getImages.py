import cv2
import  glob
import re

cap = cv2.VideoCapture("http://[2a00:20:6040:4c8e::d1]:8080/video")

#cap = cv2.VideoCapture("http://[2a00:20:6055:ca72::53]:8080/video")
images = glob.glob('images/*.png')

if len(images) >0:
    num = int(re.findall(pattern=r"\d+",string=images[-1].strip(".png"))[0]) +1
else:
    num = 0


while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()