Introduction
This repository provides an open-source Synthetic Aperture Radar (SAR) echo dataset, along with the corresponding Python code for SAR imaging/visualization (Range-Doppler Algorithm) and deep learning dataset construction. It is designed to help researchers and developers working on SAR signal processing, imaging algorithms, or deep learning-based SAR applications.

📥 Data Download
The raw SAR echo data (12 scenes in total) is hosted on Baidu Netdisk.
Link:  https://pan.baidu.com/s/1l2x9R4EeaVNwsp-VlP-PIw?pwd=edy8

Dataset Split:
Echo 1 - 6: Training data.
Echo 7 - 10: Validation data (Captured with the same system parameters as the training data).
Echo 11 - 12: Testing data (Captured with different system parameters to evaluate algorithm generalization).

⚙️ System Parameters (For Echoes 1 - 10)
The table below details the SAR system configurations used to generate the training and validation datasets.
Configuration	System parameters
Platform Height	10000 m
Platform velocity	400 m/s
Center incidence angle	60°
Sampling Rate	192 MHz
Pulse Time Width	2.0 us
Bandwidth	160 MHz
PRF (Pulse Repetition Frequency)	480 Hz
Carrier Frequency	10 GHz

💻 Code Features
step0_show_large_image.py: Reads raw .dat echo files, parses the header parameters, and performs SAR imaging using the Range-Doppler (RD) Algorithm. It provides a clear visual comparison between a basic compression result and a full-process result (including SRC - Secondary Range Compression and RCMC - Range Cell Migration Correction).
step1_make_dataset.py: Processes the complex SAR images into training datasets suitable for Deep Learning. It applies sliding window cropping (256x256 patches), separates the Real and Imaginary parts of the complex data into channels (2, H, W), and saves them as .npy arrays.

🛠️ Repository Structure
├── data/                         # Folder to store the downloaded .dat files
├── step0_show_large_image.py     # SAR RD imaging and visualization script
├── step1_make_dataset.py         # Dataset creation script for Deep Learning
├── sar_core.py                   # (Ensure you upload this core file)
├── requirements.txt              # Python dependencies
└── README.md

🚀 Getting Started
1.Clone this repository.
2.Download the 12 echo data files from Baidu Netdisk and place them into the data/ directory (e.g., data/Echo_Scene5.dat).
3.Install required libraries: pip install numpy matplotlib tqdm
4.Visualize the SAR data:
python step0_show_large_image.py
5.Build the Deep Learning Dataset:
python step1_make_dataset.py



