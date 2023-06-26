import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import time
from peakutils import indexes

phone = False
if phone:
    cv2.VideoCapture("http://[2a00:20:6006:f9aa::b3]:8080/video")
else:
    cap = cv2.VideoCapture(r"C:\Users\o.abdulmalik\PycharmProjects\image\Script\test_files\VID20230621154926.mp4")

camera_mat = np.load(r'/Script/calibration_parameters/camera_matrix.npy')
dist_mat = np.load(r'/Script/calibration_parameters/dist.npy')

x_coords = {i: [] for i in range(1,     213)}
pitch_dict = {i: [] for i in range(1,   213)}
roll_dict = {i: [] for i in range(1,    213)}
yaw_dict = {i: [] for i in range(1,     213)}


y_coords = {i: [] for i in range(1,         213)}
timestamps     =  {i: [] for i in range(1,  213)}
velocities = {i: [] for i in range(1,       213)}

fps_start_time = time.time()
fps_frame_counter = 0


def calculate_frequency(input_signal, sampling_rate):
    # Detect peaks in the input signal
    peak_indices = indexes(input_signal)

    # Calculate the peak-to-peak distances
    peak_distances = np.diff(peak_indices)

    # Average the peak distances
    average_peak_distance = np.mean(peak_distances)

    # Calculate the frequency as the reciprocal of the average peak distance
    frequency = 1 / (average_peak_distance / sampling_rate)

    return frequency

def draw_corner_dots(image, corners):
    """
    Draw green dots on the corners of markers.

    Args:
        image (numpy.ndarray): Input image.
        corners (numpy.ndarray): Marker corners.

    Returns:
        numpy.ndarray: Image with green dots on corners.
    """
    for i, marker_corners in enumerate(corners):
        for j, corner in enumerate(marker_corners):
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Draw a green dot on each corner
            #cv2.putText(image, f'{j + 1}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Put corner number
    return image

def scale_length(length, measured_perimeter, desired_perimeter):
    """
    Scale a length based on measured and desired perimeters.

    Args:
        length (float): Length to be scaled.
        measured_perimeter (float): Measured perimeter.
        desired_perimeter (float): Desired perimeter.

    Returns:
        float: Scaled length.
    """
    scaling_factor = desired_perimeter / measured_perimeter
    scaled_length = length * scaling_factor
    return scaled_length

def track_center_coordinates(corners, ids):
    """
    Track the center coordinates of markers and their corresponding timestamps.

    Args:
        corners (numpy.ndarray): Array of marker corners.
        ids (numpy.ndarray): Array of marker IDs.
        timestamps (dict): Dictionary to store the timestamps for each marker.

    Returns:
        None

    """
    for marker_index, marker_id in enumerate(ids):
        marker_id = int(marker_id)

        if marker_id in ids:
            center = (
                int((corners[marker_index][0][0][0] + corners[marker_index][0][2][0]) / 2),
                int((corners[marker_index][0][0][1] + corners[marker_index][0][2][1]) / 2)
            )
            x_coords[marker_id].append(center[0])
            y_coords[marker_id].append(center[1])
            timestamps[marker_id].append(time.time())
        else:
            if len(y_coords[marker_id]) > 0:
                x_coords[marker_id].append(x_coords[marker_id][-1])
                y_coords[marker_id].append(y_coords[marker_id][-1])






def connecting_line(index1,index2,ids,frame,dicts):
    if index1 in ids and index2 in ids:
        index_10 = list(ids).index(index1)
        index_11 = list(ids).index(index2)
        center_10 = (int((corners[index_10][0][0][0] + corners[index_10][0][2][0]) / 2),
                     int((corners[index_10][0][0][1] + corners[index_10][0][2][1]) / 2))
        center_11 = (int((corners[index_11][0][0][0] + corners[index_11][0][2][0]) / 2),
                     int((corners[index_11][0][0][1] + corners[index_11][0][2][1]) / 2))
        cv2.line(frame, center_10, center_11, (0, 0, 255), 2)

        length = (((center_11[0] - center_10[0]) ** 2 + (center_11[1] - center_10[1]) ** 2) ** 0.5)
        scale = (dicts[index2]+dicts[index2])//2
        length = scale_length(length=length,measured_perimeter=scale,desired_perimeter=108)
        length_text = "L: {:.2f}".format(length)
        cv2.putText(frame, length_text, ((center_10[0]+center_11[0])//2, (center_10[1]+center_11[1])//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
try:
    fps = 0
    fpsl = []
    while True:
        ret, frame = cap.read()
        #frame = cv2.undistort(frame, cameraMatrix=camera_mat, distCoeffs=dist_mat)
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters,)

        fps_frame_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_frame_counter / (time.time() - fps_start_time)
            fps_frame_counter = 0
            fps_start_time = time.time()
        cv2.putText(frame, "FPS: {:.2f}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fpsl.append(fps)
        if len(corners) > 0:
            corners = np.array(corners, dtype=np.float32)
            reshaped_corners = corners.reshape(len(corners), 4, 2)
            dist_coeffs = np.array(dist_mat, dtype=np.float32)
            camera_matrix = np.array(camera_mat[0], dtype=np.float32)
            marker_length = 27

            perim_dict = {ids[i][0]: 0 for i in range(len(corners))}
            perimeters = []
            for i in range(len(corners)):
                # Calculate center coordinates
                center_x = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
                center_y = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
                center = (center_x, center_y)

                # Draw center point on the frame
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                perimeters.append(cv2.arcLength(corners[i], True))

                int_corners = np.int64(corners[i])

                draw_corner_dots(image=frame, corners=int_corners)

                cv2.polylines(frame, int_corners, True, (255, 255, 0), 2)
                perim_dict[ids[i][0]] = float(cv2.arcLength(corners[i], True))
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=corners[i], distCoeffs=dist_coeffs,
                                                                               cameraMatrix=camera_mat, markerLength=0.027)

                cv2.drawFrameAxes(frame, cameraMatrix=camera_mat, distCoeffs=dist_coeffs, rvec=rvec, tvec=tvec,
                                  length=0.012)
                perimeter = np.mean(perimeters)
                # Reshape tvec and rvec to remove the extra dimension
                tvec = np.reshape(tvec, (3,))
                rvec = np.reshape(rvec, (3,))

                ## Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Create the 3x4 projection matrix
                projection_matrix = np.hstack((rotation_matrix, tvec[:, np.newaxis]))

                # Decompose the projection matrix
                retval, out_camera_matrix, out_rot_matrix, out_translation_vector, out_rot_matrixx, out_rot_matrixy, out_rot_matrixz = cv2.decomposeProjectionMatrix(projection_matrix)

                # Extract tilt angles (pitch and roll) from the rotation matrix
                pitch = np.degrees(np.arcsin(-out_rot_matrix[2, 0]))
                roll = np.degrees(np.arctan2(out_rot_matrix[1, 0], out_rot_matrix[0, 0]))
                try:
                    pitch_dict[ids[i][0]].append(pitch)
                    roll_dict[ids[i][0]].append(roll)
                except Exception as e:
                    print(ids)
                    print(e)


            track_center_coordinates(corners=corners, ids=ids)
            aruco.drawDetectedMarkers(frame, corners, ids)




        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(e)
cap.release()
cv2.destroyAllWindows()

fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(8, 8))
sampling_rate = np.mean(fpsl)
print(sampling_rate)
for key in y_coords.keys():
    if len(y_coords[key]) > 0 and np.mean(y_coords[key]) > 0:
        x = (x_coords[key] / np.max(x_coords[key]))
        x -=  np.mean(x)
        x*= 10
        print(calculate_frequency(sampling_rate=sampling_rate, input_signal=x_coords[3]))
        print(calculate_frequency(sampling_rate=sampling_rate, input_signal=x_coords[4]))
        ax1.plot(timestamps[key], x, label=f"X:{key}")
        ax2.plot(timestamps[key], y_coords[key], label=f"Y:{key}")
        ax3.plot(timestamps[key], pitch_dict[key], label=f"Pitch:{key}")
        ax4.plot(timestamps[key], roll_dict[key], label=f"Roll:{key}")



#        ax5.plot(timestamps[key], yaw_dict[key], label=f"Yaw:{key}")


ax1.set_ylabel('Y-Coordinate')
ax1.legend()

# Plot x-coordinates


ax2.set_ylabel('X-Coordinate')
ax2.legend()
ax3.set_ylabel('Pitch')
ax3.legend()

ax4.set_ylabel('Roll')
ax4.legend()

#x5.set_ylabel('Yaw')
#x5.legend()

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Display the plot
plt.show()


