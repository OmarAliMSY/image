import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

df = pd.read_csv("test_y_coords9.csv")
print(df)
print(df["t"])
df['t'] = pd.to_datetime(df['t'], unit='s')
t = np.array(df["t"])
y = np.array((df["y"]))
y = y - np.mean(y)
y = y/np.max(y)
print()
peaks, _ = find_peaks(y, height=0,distance=15)


condition = y[peaks] > 0.05
indices = peaks[condition]
# Calculate the time differences between consecutive indices
timedeltas = []


for i in range(len(indices) - 1):
    start_index = indices[i]
    end_index = indices[i + 1]
    time_difference = (df['t'].iloc[end_index] - df['t'].iloc[start_index]).total_seconds()
    timedeltas.append(time_difference)

# Print the time differences
print(np.mean(timedeltas))

plt.plot(t[indices], y[indices], "x")
plt.hlines(0,xmin=t[0],xmax=t[-1])
plt.plot(t,y)
plt.gcf().autofmt_xdate()

plt.show()