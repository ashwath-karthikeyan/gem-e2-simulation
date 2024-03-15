import matplotlib.pyplot as plt
import numpy as np

# Total time for the simulation in seconds (2 minutes)
total_time = 120  # 2 minutes = 120 seconds

# Generate timestamps at 1-second intervals
timestamps = np.linspace(0, total_time, total_time)

# # Generate heading error values that follow a pattern of low-high-low-high
# # Using a combination of sine and cosine functions to create the pattern
# heading_error = np.sin(0.05 * timestamps) + np.cos(0.1 * timestamps) + np.random.normal(0, 0.1, total_time)

# # Modifying the pattern to make it more pronounced
# heading_error[:30] *= 0.5   # Low error in the beginning
# heading_error[30:60] *= 2   # High error
# heading_error[60:90] *= 0.5 # Low error
# heading_error[90:] *= 2     # High error at the end

# # Plotting the heading error over time
# plt.figure(figsize=(10, 6))
# plt.plot(timestamps, heading_error, label='Pure Vision Heading Error')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Heading Error')
# plt.legend()
# plt.grid(True)
# plt.show()

# Generate heading error values with overall less error and less fluctuation
# Adjusting the amplitude and frequency of the sine and cosine functions
heading_error_reduced = 0.5 * (np.sin(0.03 * timestamps) + np.cos(0.06 * timestamps) + np.random.normal(0, 0.05, total_time))

# Modifying the pattern to make the fluctuations less pronounced
heading_error_reduced[:30] *= 0.3   # Very low error in the beginning
heading_error_reduced[30:60] *= 1   # Slightly higher error
heading_error_reduced[60:90] *= 0.3 # Very low error again
heading_error_reduced[90:] *= 1     # Slightly higher error at the end

# Plotting the reduced heading error over time
plt.figure(figsize=(10, 6))
plt.plot(timestamps, heading_error_reduced, label='Vision + GNSS Heading Error', color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Heading Error')
plt.legend()
plt.grid(True)
plt.show()

