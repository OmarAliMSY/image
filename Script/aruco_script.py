import cv2
import cv2.aruco as aruco
import  numpy as np
import matplotlib.pyplot as plt
import time

phone = False

if phone:
    cap = cv2.VideoCapture("http://[2a00:20:600c:5d4::2e]:8080/video")
else:
    cap = cv2.VideoCapture(0)

#camera_mat = np.load('camera_matrix.npy')
#dist_mat = np.load('dist.npy')
x_coords = {i:[ ] for i in range(1,12)}
y_coords = {i:[ ] for i in range(1,12)}
fps_start_time = time.time()
fps_frame_counter = 0

def scale_length(length, measured_perimeter, desired_perimeter):
    scaling_factor = desired_perimeter / measured_perimeter
    scaled_length = length * scaling_factor
    return scaled_length


def connecting_line(index1,index2,ids,frame,dict,Last=False):
    if index1 in ids and index2 in ids:
        index_10 = list(ids).index(index1)
        index_11 = list(ids).index(index2)
        center_10 = (int((corners[index_10][0][0][0] + corners[index_10][0][2][0]) / 2),
                     int((corners[index_10][0][0][1] + corners[index_10][0][2][1]) / 2))
        center_11 = (int((corners[index_11][0][0][0] + corners[index_11][0][2][0]) / 2),
                     int((corners[index_11][0][0][1] + corners[index_11][0][2][1]) / 2))
        cv2.line(frame, center_10, center_11, (0, 0, 255), 2)

        length = (((center_11[0] - center_10[0]) ** 2 + (center_11[1] - center_10[1]) ** 2) ** 0.5)
        scale = (dict[index2]+dict[index2])//2
        length = scale_length(length=length,measured_perimeter=scale,desired_perimeter=108)
        length_text = "Length: {:.2f}".format(length)
        cv2.putText(frame, length_text, ((center_10[0]+center_11[0])//2, (center_10[1]+center_11[1])//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if Last :
            x_coords[index2].append(center_11[0])
            y_coords[index2].append(center_11[1])
            x_coords[index1].append(center_10[0])
            y_coords[index1].append(center_10[1])
        else:
            x_coords[index1].append(center_10[0])
            y_coords[index1].append(center_10[1])
    else:
        if len (y_coords[index1]) >0:
            x_coords[index1].append(x_coords[index1][-1])
            y_coords[index1].append(y_coords[index1][-1])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()





fps = 0
while True:
    ret, frame = cap.read()
    #frame=cv2.undistort(frame,cameraMatrix=camera_mat,distCoeffs=dist_mat)
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    fps_frame_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_frame_counter / (time.time() - fps_start_time)
        fps_frame_counter = 0
        fps_start_time = time.time()
    cv2.putText(frame, "FPS: {:.2f}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if len(corners) > 0:
        perim_dict = {ids[i][0]:0 for i in range(len(corners))}
        perimeters = []
        for i in range(len(corners)):
            # Calculate center coordinates
            center_x = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
            center_y = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
            center = (center_x, center_y)

            # Draw center point on the frame
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
        # Draw lines between marker 10 and marker 11
            perimeters.append(cv2.arcLength(corners[i],True))

            int_corners = np.int64(corners[i])
            cv2.polylines(frame,int_corners,True,(255,255,0),2)
            perim_dict[ids[i][0]] =  float(cv2.arcLength(corners[i],True))
            #cv2.putText(frame,str(float(cv2.arcLength(corners[i],True)))+",:"+str(ids[i]),(center_x+20,center_y+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        perimeter = np.mean(perimeters)
        connecting_line(1,11,ids=ids,frame=frame,dict=perim_dict)
        connecting_line(11,2,ids=ids,frame=frame,dict=perim_dict)
        connecting_line(2,9,ids=ids,frame=frame,dict=perim_dict,Last=True)
        #connecting_line(9,5,ids=ids,frame=frame,dict=perim_dict)
        #connecting_line(5,6,ids=ids,frame=frame,dict=perim_dict)
        #connecting_line(6,10,ids=ids,frame=frame,dict=perim_dict)
        #connecting_line(10,6,ids=ids ,frame=frame,dict=perim_dict)
        print(perim_dict,perimeter)






        aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()
for key in x_coords.keys():
    if len(x_coords[key])>0:
        t = np.linspace(0,len(x_coords[key]),len(x_coords[key]))
        #1plt.plot(t,x_coords[key],label=f"X:{key}")
        plt.plot(t,y_coords[key],label=f"Y:{key}")
plt.legend()
plt.show()