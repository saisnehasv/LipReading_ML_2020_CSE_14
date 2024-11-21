# 2020_CSE_14

Final Year VTU Project 

Project Name: "Intuitive Perception: Speech Recognition Using Machine Learning"

Team Members:

1. Roopashree N- 1KS17CS064

2. Sai Sneha SV- 1KS17CS070

3. Spoorthi V-   1KS17CS082

Team Name: 2020_CSE_14

Group No: G1

Project Guide: Dr. Swathi K

Department: Department of Computer Science & Engineering 

College: KS Institute of Technology 

# Intuitive Perception: Lip Reading Using Machine Learning 🎥🤖

This project focuses on lip reading by interpreting lip movements to generate text. Using deep learning techniques, particularly Convolutional Neural Networks (CNNs), the project achieves an accuracy of **73%** in interpreting lip movements on a predefined dataset.

---

## 🚀 Project Overview

Lip reading is the ability to understand spoken words by visually interpreting lip movements. This project aims to bridge communication gaps for individuals with hearing impairments or in situations where audio signals are unavailable.  

Key features include:
- Data preprocessing with facial detection and lip cropping.
- Training a CNN model for interpreting lip movements.
- Generating text predictions from video frames.

---

## 📂 Repository Structure

```plaintext
Documentation/
src/
├── data/
│   ├── raw/                 # Raw video and audio files
│   ├── processed/           # Preprocessed data (cropped lip images)
├── models/
│   ├── cnn_model.py         # CNN model implementation
│   ├── model_utils.py       # Model utility functions
├── preprocessing/
│   ├── face_detection.py    # Facial detection and lip cropping
│   ├── video_to_frames.py   # Extracting frames from videos
├── evaluation/
│   ├── metrics.py           # Accuracy and loss evaluation metrics
├── visualization/
│   ├── plot_results.py      # Visualization of results and predictions
├── main.py                  # Main script to run the project
├── README.md                # Documentation
```

## 🛠️  Tools and Technologies
**Programming Language**: Python
**Deep Learning Framework**: TensorFlow/Keras
**Libraries**: OpenCV, NumPy, Matplotlib
**Model Architecture**: Convolutional Neural Networks (CNNs)

## 📋  Setup Instructions

1. Clone the repository:
 ```bash
  git clone https://github.com/saisnehasv/LipReading_ML_2020_CSE_14.git
  cd LipReading_ML_2020_CSE_14/src
 ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the data:
- Extract frames from videos:
 ```bash
python preprocessing/video_to_frames.py --input data/raw/ --output data/processed/
 ```
- Detect faces and crop lip regions:
 ```bash
python preprocessing/face_detection.py --input data/processed/ --output data/processed/lips/
 ```
4. Train the model:
 ``` bash
python models/cnn_model.py --train data/processed/lips/ --epochs 50 --batch_size 32
 ```
5. Evaluate the model:
 ``` bash
python evaluation/metrics.py --model models/saved_model.h5 --test data/processed/lips/test/
 ```

## 🖥️ Run the project
-  Run Perception.exe to upload a video and generate a video with subtitles. 


## 📊 Model Performance
The CNN model achieved:

- Accuracy: 73% on the test dataset.
- Loss: Optimized using categorical cross-entropy.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

