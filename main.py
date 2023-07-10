from modules.aruco_tracking.track_marker import Marker


if __name__ == "__main__":
    tracker = Marker(video_source=r"C:\Users\o.abdulmalik\PycharmProjects\image\test_files\7x7markers.mp4")
    tracker.run()
    tracker.plot_data()
    tracker.save_to_csv()
