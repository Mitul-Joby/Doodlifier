# Doodlifier

Doodlifier is a program to classify doodles and alphabet drawn by the user into different classes with their corresponding confidence percentages.

<p align="center">
  <img width="500" alt="Doodle Prediction 1" src="https://user-images.githubusercontent.com/73733877/202848599-1518af7f-8ece-4b2b-9712-7510f39b0c46.png">
</p>

# Run 

- Install tensorflow
- Install requirements 
- Run `main.py`

Uses existing model and class names by default. 

To customize doodles, delete files under model and update doodle.txt with reference to available.txt, and run main.py.

# About

Doodlifier uses a Convolutional Neural Network to extract features, enhance them and then using softmax returns probabilities for each class label. An accuracy of roughly 95 percentage was attained.

<p align="center">
  <img width="500" alt="Doodle Prediction 1" src="https://user-images.githubusercontent.com/73733877/202850853-21d155c9-a942-4aa0-bdec-72a7d4e7c28e.png">
</p>

### [Doodle](generate/doodle.py)
- From [doodle.txt](doodle.txt), the required class npy files are downloaded from Google's Quickdraw datasets. 
  
  Check [available.txt](available.txt) for a list of available classes.
- They are randomized to try eliminating intialization bias and classes are one hot encoded and then trained and tested.
- Images are resized to 28x28 and preprocessed.
- These images are passed to the model and then class probabilities are returned.
- The model is then saved along with class names.

## Doodle Mode
<p align="center">
  <img width="300" alt="Doodle Prediction 1" src="https://user-images.githubusercontent.com/73733877/202848599-1518af7f-8ece-4b2b-9712-7510f39b0c46.png">
  &nbsp;
  <img width="300" alt="Doodle Prediction 2" src="https://user-images.githubusercontent.com/73733877/202848675-9a7576a8-c02e-47bc-b2e3-9f3f457437ef.png">
  &nbsp;
  <img width="300" alt="Doodle Prediction 3" src="https://user-images.githubusercontent.com/73733877/202848682-66bed0ab-f3ce-427e-b7e4-39a750323bc4.png">  
</p>

## Alphabet Mode
<p align="center">
  <img width="300" alt="Alphabet Prediction 1" src="https://user-images.githubusercontent.com/73733877/202848710-81f6e809-498a-44d7-958d-ff9e1945506b.png">
  &nbsp;
  <img width="300" alt="Alphabet Prediction 2" src="https://user-images.githubusercontent.com/73733877/202848728-4cc6d30d-fa7c-49fd-863d-42ac31a70cfe.png">
  &nbsp;
  <img width="300" alt="Alphabet Prediction 3" src="https://user-images.githubusercontent.com/73733877/202848817-eb53267b-71fb-47de-977b-d7462d09958e.png">  
</p>

## Datasets

Doodles
- https://github.com/googlecreativelab/quickdraw-dataset
- https://quickdraw.withgoogle.com/data/
- https://storage.googleapis.com/quickdraw_dataset/

## Refernces

- https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research#Handwriting_and_character_recognition
- https://faroit.com/keras-docs/1.2.0/
- https://github.com/tensorflow/docs
- IEEE Papers
  - https://ieeexplore.ieee.org/document/9734453
  - https://ieeexplore.ieee.org/document/9137933
  - https://ieeexplore.ieee.org/document/9537097
  - https://ieeexplore.ieee.org/document/7903730
  - https://ieeexplore.ieee.org/document/8079572
  - https://ieeexplore.ieee.org/document/8203926
  - https://ieeexplore.ieee.org/document/9213619
  - https://ieeexplore.ieee.org/document/8365201
  - https://ieeexplore.ieee.org/document/9675509


## To Do

- Muliple detections with the help of YOLO
- Better datasets for English Characters and digits
- Add more models
