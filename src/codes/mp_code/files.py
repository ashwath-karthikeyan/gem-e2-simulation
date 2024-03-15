import pandas as pd
import matplotlib.pyplot as plt

# Read the first column of each CSV file
data1 = pd.read_csv('waypts.csv', usecols=[0], header=None)
data2 = pd.read_csv('gps_values.csv', usecols=[0], header=None)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data1, label='waypts')
plt.plot(data2, label='gps')
plt.title('Comparison of First Columns')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
