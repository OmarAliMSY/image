import cv2
import glob
import pickle
import numpy as np
import re


class Calibration:
    def __init__(self, chessboard_size=(13, 9), frame_size=(1920, 1080), square_size_mm=20):
        """
        Calibration class for camera calibration and image/video processing.
        """
        self.chessboard_size = chessboard_size
        self.frame_size = frame_size
        self.square_size_mm = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objpoints = []  # 3D point in real-world space
        self.imgpoints = []  # 2D points in image plane
        self.camera_matrix = None
        self.distortion_coeffs = None

    def find_chessboard_corners(self, images_folder):
        """
        Find chessboard corners in a set of images and store object points and image points.

        Args:
            images_folder (str): Path to the folder containing calibration images.

        """
        # Prepare object points
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size_mm

        images = glob.glob(images_folder + '/*.png')

        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            # If found, add object points and image points (after refining them)
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1000)

        cv2.destroyAllWindows()

    def calibrate_camera(self):
        """
        Calibrate the camera using the collected object points and image points.
        Save the camera calibration result for later use and calculate the reprojection error.
        """
        ret, self.camera_matrix, self.distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.frame_size, None, None
        )

        # Save the camera calibration result for later use
        pickle.dump((self.camera_matrix, self.distortion_coeffs), open("calibration.pkl", "wb"))
        pickle.dump(self.camera_matrix, open("camera_matrix.pkl", "wb"))
        pickle.dump(self.distortion_coeffs, open("distortion_coeffs.pkl", "wb"))

        # Reprojection Error
        mean_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("Total error: {}".format(mean_error / len(self.objpoints)))

    def video_capture_and_save_images(self, video_source, images_folder):
        """
        Capture images from a video source and save them as individual images.

        Args:
            video_source (str or int): Video source, can be a file path or camera index.
            images_folder (str): Path to the folder to save the captured images.

        """
        cap = cv2.VideoCapture(video_source)

        images = glob.glob(images_folder + '/*.png')

        if len(images) > 0:
            num = int(re.findall(pattern=r"\d+", string=images[-1].strip(".png"))[0]) + 1
        else:
            num = 0

        while cap.isOpened():
            success, img = cap.read()
            cv2.imshow('Img', img)
            k = cv2.waitKey(5)

            if k == 27:
                break
            elif k == ord('s'):  # wait for 's' key to save and exit
                cv2.imwrite(images_folder + '/img' + str(num) + '.png', img)
                print("Image saved!")
                num += 1

        cap.release()
        cv2.destroyAllWindows()

