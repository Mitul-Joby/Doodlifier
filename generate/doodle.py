import os
import cv2
import glob
import keras
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

testImage  = 'images/doodle.png' 
modelPath  = 'models/doodle.h5'
classesPath = 'models/doodle.txt'

def load_data(root, vfold_ratio=0.2, max_items_per_class= 4000 ):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0: max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None
    
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names

x_train, y_train, x_test, y_test, class_names = load_data('data/doodle')
num_classes = len(class_names)
image_size = 28

print(len(x_train))

if __name__ == '__main__':
    index = randint(0, len(x_train))
    plt.imshow(x_train[index].reshape(28,28)) 
    plt.show()
    randomImageIndex = int(y_train[index])
    randomImageLabel = class_names[randomImageIndex]
    print(f"Random Train Image label for verification: {randomImageLabel}")

x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) 

model.build(x_train.shape)              

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['top_k_categorical_accuracy'])
print(model.summary())

model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 256, verbose=2, epochs=5)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

if __name__ == '__main__':
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

    index = randint(0, len(x_test))
    img = x_test[index]
    plt.imshow(img.squeeze()) 
    plt.show()

    pred = model.predict(np.expand_dims(img, axis=0))[0]
    top5 = (-pred).argsort()[:5]
    predictions = [(class_names[i], pred[i] * 100) for i in top5]
    print(predictions)


    test = cv2.imread(testImage, cv2.IMREAD_GRAYSCALE)
    test = cv2.resize(test, (28, 28))
    test = np.array(test)
    test = test.reshape(28, 28, 1).astype('float32')
    test /= 255.0
    test = 1 - test

    plt.imshow(test.squeeze()) 
    plt.show()

    pred = model.predict(np.expand_dims(test, axis=0))[0]
    top5 = (-pred).argsort()[:5]
    predictions = [(class_names[i], pred[i] * 100) for i in top5]
    print(predictions)

with open(classesPath, 'w') as file:
    for item in class_names:
        file.write("{}\n".format(item))

keras.models.save_model(model, modelPath)
