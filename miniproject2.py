import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points: pd.DataFrame, ecog_data: pd.Series) -> np.ndarray:
    """
    Calculate mean ERP for each finger based on trial points and ECoG data.

    Args:
        trial_points (pd.DataFrame): DataFrame with columns [start, peak, finger].
        ecog_data (pd.Series): Time-series brain signal data.

    Returns:
        np.ndarray: A 5x1201 matrix where each row corresponds to a finger (1-5) and
                    each column is the averaged brain signal for that finger.
    """
    # Initialize parameters
    num_fingers = 5  
    window_length = 1201  # 200 ms before, 1 ms start, 1000 ms after
    pre_start_offset = 200  

    # Container for ERP data for each finger
    erp_data = {finger: [] for finger in range(1, num_fingers + 1)}

    # Loop through each trial
    for _, row in trial_points.iterrows():
        start_index, _, finger = row
        # Determine the range of indices to extract
        start_window = start_index - pre_start_offset
        end_window = start_window + window_length

        # Skip trials where the range is out of bounds
        if start_window < 0 or end_window > len(ecog_data):
            continue

        # Extract the segment and append to the appropriate finger list
        segment = ecog_data[start_window:end_window].values
        erp_data[finger].append(segment)

    # Compute the mean ERP for each finger
    fingers_erp_mean = np.zeros((num_fingers, window_length))
    for finger in range(1, num_fingers + 1):
        if erp_data[finger]:  # Ensure there is data for this finger
            fingers_erp_mean[finger - 1, :] = np.mean(erp_data[finger], axis=0)

    # Plot the ERP for each finger
    plt.figure(figsize=(10, 6))
    time = np.linspace(-200, 1000, window_length)  # Time axis in ms
    for finger in range(1, num_fingers + 1):
        plt.plot(time, fingers_erp_mean[finger - 1, :], label=f'Finger {finger}')
    plt.title('Averaged Brain Response for Each Finger')
    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the matrix as a table
    fingers_erp_mean_df = pd.DataFrame(fingers_erp_mean, 
                                       index=[f'Finger {i}' for i in range(1, 6)],
                                       columns=[f'Time {i}' for i in range(1, 1202)])
    print(fingers_erp_mean_df)

    return fingers_erp_mean

# Example usage
if __name__ == "__main__":
    # Load trial points and ECoG data
    trial_points = pd.read_csv(r"C:\Users\97254\PycharmProjects\pythonProject\miniproject2\mini_project_2_data\events_file_ordered.csv", header=None, dtype=int)
    ecog_data = pd.read_csv(r"C:\Users\97254\PycharmProjects\pythonProject\miniproject2\mini_project_2_data\brain_data_channel_one.csv", header=None).squeeze()

    # Calculate mean ERP
    fingers_erp_mean = calc_mean_erp(trial_points, ecog_data)

    # Save the result if needed
    np.savetxt("fingers_erp_mean.csv", fingers_erp_mean, delimiter=",")
