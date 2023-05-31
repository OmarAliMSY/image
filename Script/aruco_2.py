import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import time

phone = False

if phone:
    cap = cv2.VideoCapture("http://[2a00:20:600c:5d4::2e]:8080/video")  # Connect to video stream from phone
else:
    cap = cv2.VideoCapture(0)  # Use default webcam

x_coords = {i: [] for i in range(1, 12)}  # Dictionary to store x-coordinates of markers
y_coords = {i: [] for i in range(1, 12)}  # Dictionary to store y-coordinates of markers
fps_start_time = time.time()  # Start time for calculating frames per second
fps_frame_counter = 0  # Counter for frames processed

def scale_length(length, measured_perimeter, desired_perimeter):
    """
    Scales a length based on the measured and desired perimeters.

    Args:
        length (float): The measured length.
        measured_perimeter (float): The measured perimeter of the object.
        desired_perimeter (float): The desired perimeter for scaling.

    Returns:
        float: The scaled length.
    """
    scaling_factor = desired_perimeter / measured_perimeter
    scaled_length = length * scaling_factor
    return scaled_length

def connecting_line(index1, index2, ids, frame, dict, Last=False):
    """
    Draws a connecting line between two markers, calculates the scaled length, and updates the coordinate dictionaries.

    Args:
        index1 (int): The index of the first marker.
        index2 (int): The index of the second marker.
        ids (numpy.ndarray): Array of marker IDs detected in the frame.
        frame (numpy.ndarray): The frame to draw on.
        dict (dict): Dictionary containing the measured perimeters for each marker.
        Last (bool): Flag indicating if it's the last marker pair.

    Returns:
        None
    """
    if index1 in ids and index2 in ids:
        index_10 = list(ids).index(index1)
        index_11 = list(ids).index(index2)
        center_10 = (int((corners[index_10][0][0][0] + corners[index_10][0][2][0]) / 2),
                     int((corners[index_10][0][0][1] + corners[index_10][0][2][1]) / 2))
        center_11 = (int((corners[index_11][0][0][0] + corners[index_11][0][2][0]) / 2),
                     int((corners[index_11][0][0][1] + corners[index_11][0][2][1]) / 2))
        cv2.line(frame, center_10, center_11, (0, 0, 255), 2)  # Draw connecting line

        length = (((center_11[0] - center_10[0]) ** 2 + (center_11[1] - center_10[1]) ** 2) ** 0.5)
        scale = (dict[index2] + dict[index2]) // 2
        length = scale_length(length=length, measured_perimeter=scale, desired_perimeter=108)  # Scale the length
        length_text = "Length: {:.2f}".format(length)
        cv2.putText(frame, length_text, ((center_10[0] + center_11[0]) // 2, (center_10[1] + center_11[1]) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Draw length text

        if Last:
            x_coords[index2].append(center_11[0])  # Update x-coordinate for marker 2
            y_coords[index2].append(center_11[1])  # Update y-coordinate for marker 2
            x_coords[index1].append(center_10[0])  # Update x-coordinate for marker 1
            y_coords[index1].append(center_10[1])  # Update y-coordinate for marker 1
        else:
            x_coords[index1].append(center_10[0])  # Update x-coordinate for marker 1
            y_coords[index1].append(center_10[1])  # Update y-coordinate for marker 1
    else:
        if len(y_coords[index1]) > 0:
            x_coords[index1].append(x_coords[index1][-1])  # Use last recorded x-coordinate
            y_coords[index1].append(y_coords[index1][-1])  # Use last recorded y-coordinate

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # Load predefined ArUco dictionary
parameters = aruco.DetectorParameters()  # Create parameters for ArUco marker detection

fps = 0  # Initialize frames per second
while True:
    ret, frame = cap.read()  # Read frame from video capture
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)  # Detect ArUco markers
    fps_frame_counter += 1  # Increment frame counter
    if (time.time() - fps_start_time) > 1:  # Calculate frames per second every second
        fps = fps_frame_counter / (time.time() - fps_start_time)
        fps_frame_counter = 0
        fps_start_time = time.time()
    cv2.putText(frame, "FPS: {:.2f}".format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Draw FPS on frame

    if len(corners) > 0:
        perim_dict = {ids[i][0]: 0 for i in range(len(corners))}  # Dictionary to store measured perimeters
        perimeters = []
        for i in range(len(corners)):
            # Calculate center coordinates
            center_x = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
            center_y = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
            center = (center_x, center_y)

            # Draw center point on the frame
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # Calculate perimeter and store in perim_dict
            perimeters.append(cv2.arcLength(corners[i], True))
            int_corners = np.int64(corners[i])
            cv2.polylines(frame, int_corners, True, (255, 255, 0), 2)
            perim_dict[ids[i][0]] = float(cv2.arcLength(corners[i], True))

        perimeter = np.mean(perimeters)  # Calculate average perimeter
        connecting_line(1, 11, ids=ids, frame=frame, dict=perim_dict)            # Draw line between marker 1 and marker 11
        connecting_line(11, 2, ids=ids, frame=frame, dict=perim_dict)            # Draw line between marker 11 and marker 2
        connecting_line(2, 9, ids=ids, frame=frame, dict=perim_dict, Last=True)  # Draw line between marker 2 and marker 9
        print(perim_dict, perimeter)

        aruco.drawDetectedMarkers(frame, corners, ids)  # Draw detected markers on frame
    cv2.imshow('Frame', frame)  # Display frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Plotting the marker coordinates over time
for key in x_coords.keys():
    if len(x_coords[key]) > 0:
        t = np.linspace(0, len(x_coords[key]), len(x_coords[key]))
        plt.plot(t, y_coords[key], label=f"Y:{key}")
plt.legend()
plt.show()
