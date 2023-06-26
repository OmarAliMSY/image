from modules.calibration.calibration import Calibration
from modules.aruco_tracking.track_marker import Marker
import glob
import  cv2 as cv

if __name__ == "__main__":
    tracker = Marker(video_source=r"C:\Users\o.abdulmalik\PycharmProjects\image\Script\test_files\7x7markers.mp4")
    tracker.run()
    tracker.plot_data()
    tracker.save_to_csv()
