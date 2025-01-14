# Emotion Recognition with Webcam

This project uses a Convolutional Neural Network (CNN) to recognize emotions from webcam images in real-time. It is trained on the FER-2013 dataset, which includes seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The model is tested with OpenCV to perform real-time emotion recognition via the webcam.

## Prerequisites

Make sure you have the following libraries installed:

- TensorFlow
- OpenCV
- NumPy

You can install the required libraries with the following commands:

```bash
pip install tensorflow opencv-python numpy
```

## Project Structure
- train.py: Script for training the emotion recognition model.
- emotion_recognition.py: Script for real-time emotion recognition using the webcam.
- model: Folder where the trained model is saved (e.g., emotion_recognition_model.h5).

## How to Train the Model
1. Prepare the data: The FER-2013 dataset is automatically loaded via the datasets library. This dataset contains images and their associated emotion labels.
2. Train the model: Use the train.py script to train the model. The dataset consists of 48x48 grayscale images, which the model uses for emotion prediction.

Example:
```bash
python train.py
```

After training, the model is saved as an .h5 file, e.g., emotion_recognition_model.h5.

## How to Test the Model with the Webcam
1. Start the webcam script: Open the emotion_recognition.py script to test the model in real-time. The script uses OpenCV to access the webcam and predict emotions from detected faces in real-time.
2. Run the test:
```bash
python emotion_recognition.py
```
This opens a window where the camera is streamed, and each detected face has its predicted emotion displayed above it. You can exit the application by pressing q.

## Code Explanation

train.py
- Loading the dataset: The FER-2013 dataset is loaded using the datasets library.
- Model Architecture: A simple Convolutional Neural Network (CNN) is created with Keras, specifically designed for emotion image classification.
- Training the model: The model is trained with the training data and validated on the test data.

emotion_recognition.py
- Webcam streaming: OpenCV is used to start the webcam and read frames continuously.
- Face detection: OpenCV’s pre-trained Haar cascades are used to detect faces in the image.
- Emotion recognition: Each detected face is preprocessed and passed to the trained model for emotion prediction.
- Displaying results: The predicted emotion is displayed on the webcam feed.

## Emotions in the FER-2013 Dataset

The model recognizes the following seven emotions:
1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

## Example Output

The application displays the webcam image with a green rectangle around the detected face. The predicted emotion is shown above the face:
```
[Happy]        [Sad]        [Neutral]
  ██████         ██████         ██████
```

## Notes
- The accuracy of emotion recognition may vary depending on facial expressions, lighting conditions, and camera quality.
- For better performance in face detection or emotion recognition, additional data preprocessing or a different face detection model may be useful.

## License
This project is licensed under the MIT License.
