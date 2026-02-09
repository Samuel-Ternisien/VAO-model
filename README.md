## Action Recognition Data Challenge
This repository contains the guidelines and documentation for the Multimodal Action Recognition Data Challenge. The goal is to classify 30 distinct action classes using a variety of sensor modalities.

# 🎯 Project Objective
The objective is to design and implement a complete machine learning pipeline—including preprocessing, modeling, and evaluation—to achieve the highest possible accuracy across 30 action classes.

Resources
Visual Documentation: View Data Visualization Video

Detailed Description: Refer to the Data Management Plan (DMP/PGD) included in the project files.

# 💾 Dataset & Modalities
Participants are provided with cleaned and synchronized multimodal data. You are free to use one or several modalities based on your strategy.

Available Modalities
IMU (Inertial Measurement Unit)

EMG (Electromyography)

Plantar Activity

Skeleton

# ⚠️ Note: The Text modality is currently unavailable and should not be used.

Data Structure & Synchronization
Because sensors operate at different frequencies, synchronization data is provided:

Event Folder: Contains timestamps used to align the different modalities temporally.

Data Path: /stockage1/mindscan/

Evaluation Split
To ensure consistency across the challenge, please use the following subject distribution: | Set | Subjects | | :--- | :--- | | Training / Validation | 1 – 24 | | Testing | 25 – 32 |

# Tools & Libraries
The server comes pre-configured with the following tools:

fisa_env: A Conda environment containing all necessary machine learning libraries.

screen: For running long-term scripts in the background.

nvtop: For real-time GPU consumption monitoring.

# 📌 Deliverables
The following items are expected for submission:

Global Strategy: A summary of your approach.

Data Processing: Details on normalization and data structuring.

Model Selection: Description of the chosen modalities and architecture(s).

Evaluation Metrics:

A confusion matrix per Subject (for subjects 25 to 32).

A confusion matrix per Action (actions 1 to 30).