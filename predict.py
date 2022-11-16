import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def predict_image(model, class_names, image):
    
    if __name__ == '__main__':
        plt.imshow(image.squeeze()) 
        plt.show()

    pred = model.predict(np.expand_dims(image, axis=0))[0]
    top5 = (-pred).argsort()[:5]
    predictions = [(class_names[i], pred[i] * 100) for i in top5]
    return predictions

if __name__ == '__main__':
    testImage  = 'images/doodle.png'
    doodleModel = load_model('models/doodle.h5')
    alpabetModel = load_model('models/alphabet.h5')
    doodleClasses = open('models/doodle.txt', 'r').read().splitlines()
    alphabetClasses = open('models/alphabet.txt', 'r').read().splitlines()

    image = cv2.imread(testImage, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = np.array(image)
    image = image.reshape(28, 28, 1).astype('float32')
    image = image / 255
    image = 1 - image

    doodlePredictions = predict_image(doodleModel, doodleClasses, image)

    print('Doodle Predictions:', doodlePredictions)
