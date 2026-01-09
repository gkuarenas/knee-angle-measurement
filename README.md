# Real-Time Knee Angle Tracker (MediaPipe + OpenCV)

## Overview
This project is a Python script that uses **MediaPipe Pose** to track the **right leg** (hip–knee–ankle) from a webcam feed and compute the **right knee angle in real time**. The script overlays the tracked joints, connecting lines, and the calculated knee angle on the video stream.

It also opens a **second camera feed** in a separate window for monitoring/recording purposes.

---

## What it does
- Captures video from **two cameras**:
  - **Camera 1:** used for pose detection + knee angle computation
  - **Camera 2:** displayed as a secondary live feed
- Detects pose landmarks using **MediaPipe Pose**
- Extracts the **RIGHT_HIP**, **RIGHT_KNEE**, and **RIGHT_ANKLE** landmarks
- Computes the knee angle using 2D geometry

## Sample image
![kneeangle](https://github.com/user-attachments/assets/9afe6912-b7c8-4453-84dd-6e93d005b9bc)
