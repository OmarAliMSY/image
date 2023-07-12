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
        self.df = pd.read_csv(self.path)
         
        df = self.df
        df['t'] = pd.to_datetime(df['t'], unit='s')

        self.t = np.array(df["t"])
        self.y = np.array(df["y"])
        self.y = self.y - np.mean(self.y)
        self.y = self.y / np.max(self.y)

        


        
        self.time_step = df['t'].diff().mean().total_seconds()
        print(self.time_step)

    def mirrored_signal(self,n):
        # Assuming self.y is your original signal
        while n >0:
            mirrored_signal = np.flip(self.y)

            # Append the mirrored signal to the original signal
            padded_signal = np.concatenate((self.y, mirrored_signal))

            self.y = padded_signal
            extended_t = np.concatenate((self.t, self.t + (self.t[-1] - self.t[0])))

            self.t = extended_t
            n-=1

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
        self.df = pd.read_csv(self.path)
        self.df['t'] = pd.to_datetime(self.df['t'], unit='s')
        self.t = np.array(self.df["t"])
        self.y = np.array((self.df["y"]))
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
        print(1/average_frequency)
        return average_frequency
    
    def calculate_fft(self):
        self.mirrored_signal(3)
        sp = np.fft.fft(np.sin(self.y))
        freq = np.fft.fftfreq(self.y.shape[-1],self.time_step)
        magnitude = np.abs(sp.real)
        
        # Apply peak detection
        peaks, _ = find_peaks(magnitude, height=30)
        sp.real = np.abs(sp.real)
        plt.plot(np.abs(freq), np.abs(sp.real))

        plt.plot(np.abs(freq[peaks]), np.abs(magnitude[peaks]), 'ro')
        plt.xlim(0,2000)
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.xlim(0,5)
        plt.show()





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
        self.mirrored_signal(3)

        peaks, _ = find_peaks(self.y, height=0, distance=1)

        condition = self.y[peaks] > 0.05
        self.indices = peaks[condition]

        
        #plt.plot(self.t[self.indices], self.y[self.indices], "x")  # Plot the peaks
        #plt.hlines(0, xmin=self.t[0], xmax=self.t[-1])  # Plot horizontal line at y=0
        plt.plot(self.t, self.y)  # Plot the oscillation data
        plt.gcf().autofmt_xdate()

        plt.show()



    def test_file(self):

        # Define parameters
        duration = 10.0  # Duration of the sine wave in seconds
        sampling_rate = 100000  # Sampling rate in Hz
        frequency = 820  # Frequency of the sine wave in Hz

        # Generate time values
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

        # Generate the sine wave
        y = np.sin(2 * np.pi * frequency * t)

        # Create a DataFrame with time and amplitude columns
        df = pd.DataFrame({'t': t, 'y': y})

        # Save DataFrame to a CSV file
        df.to_csv('test_data820hz.csv', index=False)
