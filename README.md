# Gait Features Calcualtion
This code is designed to process and analyze gait data from videos. It calculates various gait features, such as speed, body angles, stance and swing phases, cadence, asymmetry, and more. Here's a breakdown of the functionality:

**Overview**

**1.Data Loading:**

The code loads skeleton data from CSV files extracted from videos.

It calculates various features based on joint positions and timestamps in the skeleton data.


**2.Feature Calculation:**

Angles: Calculates angles of body parts with respect to horizontal and vertical axes.

Speed: Determines the speed of walking.

Stance and Swing Phases: Measures the duration of stance and swing phases of the gait cycle.

Cadence: Calculates the walking cadence (steps per minute).

Asymmetry: Computes the asymmetry between left and right leg movements in terms of stance, swing, peak, and bottom amplitudes.

Step and Stride Length: Measures the step and stride lengths.

Falling Risk: Estimates the risk of falling based on body posture.

**3.Statistical Analysis:**

Conducts a t-test to compare normal and abnormal gait features.

Generates box plots for visual comparison of gait features between normal and abnormal gaits.
