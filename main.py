from modules.aruco_tracking.track_marker import Marker
from modules.frequency.Oscilation import Frequency

path = "_data/test_y_coords9.csv"

if __name__ == "__main__":
    #tracker = Marker(video_source=r"C:\Users\o.abdulmalik\PycharmProjects\image\test_files\7x7markers.mp4")
    #tracker.run()
    #tracker.plot_data()
    #tracker.save_to_csv()
    freq = Frequency(path=path)
    #freq.calculate_frequency()
    #freq.calculate_fft()
    freq.plot_oscillation()
