import os
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.signal import find_peaks


class Marker:
    """
    Class for marker detection and tracking.

    Args:
        video_source (str): Path or index of the video source.

    Attributes:
        perim_dict (dict): Dictionary to store marker perimeters.
        x_coords (dict): Dictionary to store X coordinates of marker centers.
        y_coords (dict): Dictionary to store Y coordinates of marker centers.
        pitch_dict (dict): Dictionary to store pitch angles of markers.
        roll_dict (dict): Dictionary to store roll angles of markers.
        yaw_dict (dict): Dictionary to store yaw angles of markers.
        timestamps (dict): Dictionary to store timestamps of marker detections.
        fps_start (float): Start time for calculating FPS.
        fps_frame_counter (int): Counter for frames used in FPS calculation.
        fpsl (list): List to store calculated FPS values.
        frame (numpy.ndarray): Current frame from the video source.
        corners (numpy.ndarray): Array of detected marker corners.
        marker_perimeter (int): Perimeter of the marker used for scaling measurements.
        ids (numpy.ndarray): Array of detected marker IDs.
        aruco_dict: ArUco dictionary for marker detection.
        parameters: ArUco detector parameters.
        marker_length (float): Length of the marker in meters.
        camera_mat (numpy.ndarray): Camera matrix for calibration.
        dist_mat (numpy.ndarray): Distortion matrix for calibration.
        cap: VideoCapture object for reading video frames.

    """

    def __init__(self,video_source, aruco_dict=aruco.DICT_7X7_250):
        self.perim_dict = {}
        self.x_coords = {i: [] for i in range(1, 213)}
        self.y_coords = {i: [] for i in range(1, 213)}
        self.pitch_dict = {i: [] for i in range(1, 213)}
        self.roll_dict = {i: [] for i in range(1, 213)}
        self.yaw_dict = {i: [] for i in range(1, 213)}
        self.timestamps = {i: [] for i in range(1, 213)}
        self.fps_start = time.time()
        self.fps_frame_counter = 0
        self.fpsl = []
        self.frame = None
        self.corners = None
        self.marker_perimeter = 108
        self.ids = None
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = aruco.DetectorParameters()
        self.marker_length = 27
        self.camera_mat = np.load(r'calibration_parameters/camera_matrix.npy')
        self.dist_mat = np.load(r'calibration_parameters/dist.npy')
        self.cap = cv2.VideoCapture(video_source)

    def corner_dots(self):
        """
        Draw circles on the detected marker corners.

        Returns:
            None
        """
        for corners in self.corners:
            for corner in corners:
                x, y = corner[0]
                x = int(x)
                y = int(y)
                cv2.circle(self.frame, (x, y), 4, (0, 0, 255), -1)

    def scale_length(self, length, measured_perimeter):
        """
        Scale a length based on measured and desired perimeters.

        Args:
            length (float): Length to be scaled.
            measured_perimeter (float): Measured perimeter.

        Returns:
            float: Scaled length.
        """
        scaling_factor = self.marker_perimeter / measured_perimeter
        scaled_length = length * scaling_factor
        return scaled_length

    def track_center_coordinates(self):
        """
        Track the center coordinates of markers and their corresponding timestamps.

        Returns:
            None
        """
        if self.ids is not None:
            for marker_index, marker_id in enumerate(self.ids):
                marker_id = int(marker_id)

                if marker_id in self.ids:
                    center = (
                        int((self.corners[marker_index][0][0][0] + self.corners[marker_index][0][2][0]) / 2),
                        int((self.corners[marker_index][0][0][1] + self.corners[marker_index][0][2][1]) / 2)
                    )
                    self.x_coords[marker_id].append(center[0])
                    self.y_coords[marker_id].append(center[1])
                    self.timestamps[marker_id].append(time.time())
                else:
                    if len(self.y_coords[marker_id]) > 0:
                        self.x_coords[marker_id].append(self.x_coords[marker_id][-1])
                        self.y_coords[marker_id].append(self.y_coords[marker_id][-1])

    def connecting_line(self, index1, index2):
        """
        Draw a connecting line between two markers.

        Args:
            index1 (int): ID of the first marker.
            index2 (int): ID of the second marker.

        Returns:
            None
        """
        for i in range(len(self.corners)):
            center_x = int((self.corners[i][0][0][0] + self.corners[i][0][2][0]) / 2)
            center_y = int((self.corners[i][0][0][1] + self.corners[i][0][2][1]) / 2)
            center = (center_x, center_y)
            cv2.circle(self.frame, center, 5, (0, 255, 0), -1)
            self.perim_dict[self.ids[i][0]] = float(cv2.arcLength(self.corners[i], True))
        if self.ids is not None:
            if index1 in self.ids and index2 in self.ids:
                index_10 = list(self.ids).index(index1)
                index_11 = list(self.ids).index(index2)
                center_10 = (int((self.corners[index_10][0][0][0] + self.corners[index_10][0][2][0]) / 2),
                             int((self.corners[index_10][0][0][1] + self.corners[index_10][0][2][1]) / 2))
                center_11 = (int((self.corners[index_11][0][0][0] + self.corners[index_11][0][2][0]) / 2),
                             int((self.corners[index_11][0][0][1] + self.corners[index_11][0][2][1]) / 2))
                cv2.line(self.frame, center_10, center_11, (0, 0, 255), 2)

                length = (((center_11[0] - center_10[0]) ** 2 + (center_11[1] - center_10[1]) ** 2) ** 0.5)
                scale = (self.perim_dict[index1] + self.perim_dict[index2]) // 2
                length = self.scale_length(length=length, measured_perimeter=scale)
                length_text = "L: {:.2f}".format(length)
                cv2.putText(self.frame, length_text,
                            ((center_10[0] + center_11[0]) // 2, (center_10[1] + center_11[1]) // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def calculate_pose(self):
        """
        Calculate the pose (position and orientation) of the markers.

        Returns:
            None
        """
        for i in range(len(self.corners)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners=self.corners[i],
                distCoeffs=self.dist_mat,
                cameraMatrix=self.camera_mat,
                markerLength=0.027
            )

            cv2.drawFrameAxes(self.frame, cameraMatrix=self.camera_mat, distCoeffs=self.dist_mat, rvec=rvec, tvec=tvec,
                              length=0.012)

            tvec = np.reshape(tvec, (3,))
            rvec = np.reshape(rvec, (3,))
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            projection_matrix = np.hstack((rotation_matrix, tvec[:, np.newaxis]))
            retval, out_camera_matrix, out_rot_matrix, out_translation_vector, out_rot_matrixx, out_rot_matrixy, out_rot_matrixz = cv2.decomposeProjectionMatrix(
                projection_matrix)
            pitch = np.degrees(np.arcsin(-out_rot_matrix[2, 0]))
            roll = np.degrees(np.arctan2(out_rot_matrix[1, 0], out_rot_matrix[0, 0]))
            self.pitch_dict[self.ids[i][0]].append(pitch)
            self.roll_dict[self.ids[i][0]].append(roll)

    def plot_data(self):
        """
        Plot the tracked data.

        Returns:
            None
        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))
        sampling_rate = np.mean(self.fpsl)
        print(sampling_rate)
        for key in self.y_coords.keys():
            if len(self.y_coords[key]) > 0 and np.mean(self.y_coords[key]) > 0:
                x = (self.x_coords[key] / np.max(self.x_coords[key]))
                x -= np.mean(x)
                ax1.plot(self.timestamps[key], x, label=f"X:{key}")
                ax2.plot(self.timestamps[key], self.y_coords[key], label=f"Y:{key}")
                ax1.axhline(y=0)
                ax3.plot(self.timestamps[key], self.pitch_dict[key], label=f"Pitch:{key}")
                ax4.plot(self.timestamps[key], self.roll_dict[key], label=f"Roll:{key}")
                #self.calculate_oscillation_frequency(self.y_coords[key])
                a = np.array(self.x_coords[key])
                t = np.array(self.timestamps[key])


        ax1.set_ylabel('Y-Coordinate')
        ax1.legend()

        ax2.set_ylabel('X-Coordinate')
        ax2.legend()

        ax3.set_ylabel('Pitch')
        ax3.legend()

        ax4.set_ylabel('Roll')
        ax4.legend()
        plt.subplots_adjust(hspace=0.4)

        plt.show()



    def save_to_csv(self,directory = os.getcwd()):


        for key in self.y_coords.keys():
            if len(self.y_coords[key]) > 0 and np.mean(self.y_coords[key]) > 0:
                x = (self.x_coords[key] / np.max(self.x_coords[key]))
                x -= np.mean(x)
                a = np.array(self.x_coords[key])
                t = np.array(self.timestamps[key])
                df = pd.DataFrame({"t": t, "y": a})
                # Speichere das DataFrame als CSV-Datei
                df.to_csv(directory+"/test_y_coords" + str(key) + ".csv", index=False)

    def frequencies(self,time, coords):
        t = time
        y = np.array(coords)
        y = y - np.mean(y)
        y = y / np.max(y)
        peaks, _ = find_peaks(y, height=0, distance=15)

        indices = peaks[1:]
        # Calculate the time differences between consecutive indices
        timedeltas = []
        for i in range(len(indices) - 1):
            start_index = indices[i]
            end_index = indices[i + 1]
            time_difference = t[end_index] - t[start_index]
            timedeltas.append(time_difference)

        # Convert timedeltas to seconds
        timedeltas = np.array(timedeltas).astype(np.float64)  # Convert to NumPy float64 array
        timedeltas = timedeltas * 1e-9  # Convert nanoseconds to seconds

        # Calculate the frequencies from the time differences
        frequencies = 1 / np.mean(timedeltas)

        return frequencies

    def run(self):
        """
        Run the marker tracking and pose estimation.

        Returns:
            None
        """
        fps = 0
        while True:
            try:
                ret, self.frame = self.cap.read()
                self.corners, self.ids, rejected = aruco.detectMarkers(self.frame, self.aruco_dict,
                                                                        parameters=self.parameters)

                self.fps_frame_counter += 1
                if (time.time() - self.fps_start) > 1:
                    self.fpsl.append(self.fps_frame_counter / (time.time() - self.fps_start))
                    self.fps_start = time.time()
                    self.fps_frame_counter = 0

                self.track_center_coordinates()
                self.calculate_pose()

                self.corner_dots()
                self.connecting_line(3,4)

                cv2.imshow('Frame', self.frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(e)
                break
        self.cap.release()
        cv2.destroyAllWindows()
