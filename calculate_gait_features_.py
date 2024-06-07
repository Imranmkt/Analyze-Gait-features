
#the following code is optimized version of original code.Original code credit goes to my labmate Xinyi

import os
import pandas as pd
import numpy as np
from math import ceil, atan, degrees
import json

from pyts.preprocessing import InterpolationImputer
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import stats

from utils import calculate_angles, separate_normal_abnormal, separate_template, generate_distance_DTW

VIDEO_ROOT_DIR = "your_path_directory"
selected_folders = ["file-1", "file-2", "file-3", "X"]

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

def calculate_horizontal_angles(joint1, joint2):
    """Calculate the angle between two joints with respect to the horizontal axis."""
    x1, y1 = joint1
    x2, y2 = joint2
    delta_x = abs(x1 - x2)
    delta_y = abs(y1 - y2)

    if y1 == y2:
        return 0

    angle = atan(delta_y / delta_x)
    angle = degrees(angle)

    return angle if y1 < y2 else -angle

def load_skeleton_data(video_root_dir, folder, video_file):
    """Load skeleton data from a given video and calculate various gait features."""
    file_dir = os.path.join(video_root_dir, folder, video_file[:-4])
    joint_data = pd.read_csv(os.path.join(file_dir, "jointdata.csv"))
    joint_data["data"] = joint_data["data"].apply(json.loads)
    index = pd.read_csv(file_dir + "primer.csv", header=None)[0][0]

    index_1 = len(joint_data) // 3
    index_2 = len(joint_data) // 3 * 2

    # Determine walking direction
    if joint_data.iloc[index_1]["data"][8][0] > joint_data.iloc[index_2]["data"][8][0]:
        direction = "L"
    else:
        direction = "R"

    # Flip the x coordinates if walking towards left
    if direction == "L":
        for i in range(len(joint_data)):
            for j in range(25):
                joint_data["data"][i][j][0] = IMAGE_WIDTH - joint_data["data"][i][j][0]

    # Calculate speed in pixels per second
    neck_1 = joint_data["data"][index_1][1]
    neck_2 = joint_data["data"][index_2][1]
    mid_hip_1 = joint_data["data"][index_1][8]
    mid_hip_2 = joint_data["data"][index_2][8]
    delta_x = (neck_2[0] + mid_hip_2[0]) / 2 - (neck_1[0] + mid_hip_1[0]) / 2
    delta_t = abs(joint_data["time"][index_2] - joint_data["time"][index_1]) / 1000

    if delta_t == 0:
        print("Error in speed calculation")
        return -1
    else:
        speed = abs(delta_x / delta_t)

    # Calculate angles
    neck = joint_data["data"][index][1]
    mid_hip = joint_data["data"][index][8]
    nose = joint_data["data"][index][0]

    body_lean = calculate_angles(nose, mid_hip)
    back_lean = calculate_angles(neck, mid_hip)
    neck_lean = calculate_angles(nose, neck)

    # Determine valid skeleton data range
    idx_start = int(round(joint_data["time"][0] * 30 / 1000))
    idx_end = int(round(joint_data["time"][joint_data.shape[0] - 1] * 30 / 1000))
    seq_len = idx_end - idx_start + 1

    left_seq = [None] * seq_len
    right_seq = [None] * seq_len
    index_seq = [None] * seq_len

    for i in range(joint_data.shape[0]):
        original_index = int(round(joint_data["time"][i] * 30 / 1000))
        left_seq[original_index - idx_start] = calculate_angles(joint_data["data"][i][13], joint_data["data"][i][14])
        right_seq[original_index - idx_start] = calculate_angles(joint_data["data"][i][10], joint_data["data"][i][11])
        index_seq[original_index - idx_start] = i

    imputer = InterpolationImputer()
    impute_index = list(range(idx_start, idx_end + 1))
    left = np.array(imputer.transform([impute_index, left_seq])[1])
    right = np.array(imputer.transform([impute_index, right_seq])[1])

    peaks_left, bottoms_left = find_sequence_extrema(left)
    peaks_right, bottoms_right = find_sequence_extrema(right)

    left_stance, left_swing = calculate_stance_swing(phases(peaks_left, bottoms_left))
    right_stance, right_swing = calculate_stance_swing(phases(peaks_right, bottoms_right))

    asymmetry_stance_phase = calculate_asymmetry(left_stance, right_stance)
    asymmetry_swing_phase = calculate_asymmetry(left_swing, right_swing)

    cadence = calculate_cadence(peaks_left, peaks_right)

    left_peak_amp = np.mean([left[i] for i in peaks_left])
    left_bottom_amp = np.mean([left[i] for i in bottoms_left])
    right_peak_amp = np.mean([right[i] for i in peaks_right])
    right_bottom_amp = np.mean([right[i] for i in bottoms_right])

    asymmetry_peak_amplitude = calculate_asymmetry(left_peak_amp, right_peak_amp)
    asymmetry_bottom_amplitude = calculate_asymmetry(left_bottom_amp, right_bottom_amp)

    left_index = ceil(len(peaks_left) / 2) - 1
    right_index = ceil(len(peaks_right) / 2) - 1

    left_step_length = abs(joint_data["data"][index_seq[peaks_left[left_index]]][21][0] -
                           joint_data["data"][index_seq[peaks_left[left_index]]][24][0])
    right_step_length = abs(joint_data["data"][index_seq[peaks_left[right_index]]][21][0] -
                            joint_data["data"][index_seq[peaks_left[right_index + 1]]][24][0])

    asymmetry_step_length = calculate_asymmetry(left_step_length, right_step_length)

    stride_length = abs(joint_data["data"][index_seq[peaks_left[left_index]]][21][0] -
                        joint_data["data"][index_seq[peaks_right[left_index]]][21][0])

    falling_risk = abs(joint_data["data"][index_seq[peaks_left[left_index]]][0][0] - (
            joint_data["data"][index_seq[peaks_left[left_index]]][21][0] +
            joint_data["data"][index_seq[peaks_left[left_index]]][24][0]) / 2) / (
                           abs(joint_data["data"][index_seq[peaks_left[left_index]]][19][0] -
                               joint_data["data"][index_seq[peaks_left[left_index]]][24][0]) / 2)

    features = [speed, body_lean, back_lean, neck_lean,
                left_stance, left_swing, right_stance, right_swing, asymmetry_stance_phase, asymmetry_swing_phase,
                cadence, left_peak_amp, right_peak_amp,
                left_bottom_amp, right_bottom_amp, asymmetry_peak_amplitude, asymmetry_bottom_amplitude,
                left_step_length, right_step_length, asymmetry_step_length,
                stride_length, falling_risk]

    return features

def find_sequence_extrema(sequence):
    """Find peaks and bottoms in a given sequence."""
    peaks, _ = find_peaks(sequence, prominence=(30, None))
    bottoms, _ = find_peaks(-sequence, prominence=(30, None))

    if bottoms[0] < peaks[0]:
        bottoms = bottoms[1:]
    if bottoms[-1] > peaks[-1]:
        bottoms = bottoms[:-1]

    return peaks, bottoms

def calculate_stance_swing(phases):
    """Calculate stance and swing phases from peaks and bottoms."""
    peaks, bottoms = phases

    if len(peaks) - len(bottoms) != 1:
        print("Missing or incorrect data")

    stance_phases = [bottoms[i] - peaks[i] for i in range(len(bottoms))]
    swing_phases = [peaks[i + 1] - bottoms[i] for i in range(len(bottoms))]

    return np.mean(stance_phases) / 30, np.mean(swing_phases) / 30

def calculate_cadence(peaks_left, peaks_right):
    """Calculate cadence from left and right peaks."""
    if len(peaks_left) != len(peaks_right):
        print("Mismatch in peaks length")
        min_len = min(len(peaks_left), len(peaks_right))
        peaks_left, peaks_right = peaks_left[:min_len], peaks_right[:min_len]

    step_times = [(peaks_left[i] - peaks_right[i]) for i in range(len(peaks_right))]
    avg_step_time = np.mean(step_times) / 30

    return 60 / avg_step_time

def calculate_asymmetry(value_a, value_b):
    """Calculate the asymmetry between two values."""
    return abs(value_a - value_b) / max(value_a, value_b)

def main():
    """Main function to process videos and extract gait features."""
    for folder in selected_folders:
        file_list = os.listdir(os.path.join(VIDEO_ROOT_DIR, folder))
        normal_videos, abnormal_videos = separate_normal_abnormal(file_list)

        X_normal = [load_skeleton_data(VIDEO_ROOT_DIR, folder, video) for video in normal_videos]
        X_abnormal = [load_skeleton_data(VIDEO_ROOT_DIR, folder, video) for video in abnormal_videos]

        Y_normal = ["normal"] * len(normal_videos)
        Y_abnormal = ["abnormal"] * len(abnormal_videos)

        X = np.concatenate((X_normal, X_abnormal), axis=0)
        Y = np.concatenate((Y_normal, Y_abnormal), axis=0)

        data = pd.DataFrame(X, columns=["Speed", "Body Lean", "Back Lean", "Neck Lean",
                                        "Left Stance", "Left Swing", "Right Stance", "Right Swing", "Asymmetry Stance Phase", "Asymmetry Swing Phase",
                                        "Cadence", "Left Peak Amplitude", "Right Peak Amplitude",
                                        "Left Bottom Amplitude", "Right Bottom Amplitude", "Asymmetry Peak Amplitude", "Asymmetry Bottom Amplitude",
                                        "Left Step Length", "Right Step Length", "Asymmetry Step Length",
                                        "Stride Length", "Falling Risk"])
        data["Type"] = pd.Series(Y)

        data.to_excel(folder + '.xlsx')

        normal = data[data["Type"] == "normal"]
        abnormal = data[data["Type"] == "abnormal"]

        box_plot_dir = os.path.join(VIDEO_ROOT_DIR, folder, "box_plot")
        if not os.path.exists(box_plot_dir):
            os.makedirs(box_plot_dir)

        for feature in data.columns:
            if feature == "Type":
                continue
            print(f"--------------------------\n{feature}\n{stats.ttest_ind(normal[feature], abnormal[feature])}\n--------------------------")

            fig, ax = plt.subplots()
            ax.set_title(feature)
            ax.set_xticklabels(['Normal Gait', 'Abnormal Gait'])
            ax.boxplot([normal[feature], abnormal[feature]])
            plt.savefig(os.path.join(box_plot_dir, feature + ".png"))
            plt.show()

if __name__ == '__main__':
    main()
