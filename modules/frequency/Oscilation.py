import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


class Frequency:
    """
    Class for calculating frequency and plotting oscillation data.
    """

    def __init__(self, path):
        """
        Initialize the Frequency object.

        Args:
            path (str): Path to the input CSV file containing time and amplitude data.
        """
        self.path = path
        self.t = None
        self.y = None
        self.indices = None

    def calculate_frequency(self):
        """
        Calculate the frequency of oscillation.

        Returns:
            float: The calculated frequency.

        Notes:
            - This method reads the data from the input CSV file.
            - It performs peak detection and calculates the time differences between consecutive peaks.
            - The average time difference is used to calculate the frequency.
        """
        df = pd.read_csv(self.path)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        self.t = np.array(df["t"])
        self.y = np.array((df["y"]))
        self.y = self.y - np.mean(self.y)
        self.y = self.y / np.max(self.y)

        peaks, _ = find_peaks(self.y, height=0, distance=15)

        condition = self.y[peaks] > 0.05
        self.indices = peaks[condition]

        # Calculate the time differences between consecutive indices
        timedeltas = []
        for i in range(len(self.indices) - 1):
            start_index = self.indices[i]
            end_index = self.indices[i + 1]
            time_difference = (df['t'].iloc[end_index] - df['t'].iloc[start_index]).total_seconds()
            timedeltas.append(time_difference)

        # Print the average time difference (frequency)
        average_frequency = np.mean(timedeltas)
        print(average_frequency)
        return average_frequency

    def plot_oscillation(self):
        """
        Plot the oscillation data.
    
        This method checks if the time (`self.t`) and amplitude (`self.y`) arrays are None.
        If they are None, the method calls the `calculate_frequency()` method to populate them and calculate the indices of peaks.
    
        The oscillation data is then plotted with marked peaks.
    
        Notes:
            - If the time and amplitude arrays are already populated, the method directly proceeds to plotting.
            - The peaks are marked as 'x' symbols in the plot.
            - A horizontal line at y=0 is plotted for reference.
        """

        if self.t is None and self.y is None:
            self.calculate_frequency()

        plt.plot(self.t[self.indices], self.y[self.indices], "x")  # Plot the peaks
        plt.hlines(0, xmin=self.t[0], xmax=self.t[-1])  # Plot horizontal line at y=0
        plt.plot(self.t, self.y)  # Plot the oscillation data
        plt.gcf().autofmt_xdate()

        plt.show()
