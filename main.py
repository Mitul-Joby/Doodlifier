import sys
import cv2
import pygame
import numpy as np
from ctypes import windll
from predict import predict_image
from keras.models import load_model

models = {
    'doodle': load_model('models/doodle.h5'),
    'alphabet': load_model('models/alphabet.h5')
}
classes = {
    'doodle': open('models/doodle.txt', 'r').read().splitlines(),
    'alphabet': open('models/alphabet.txt', 'r').read().splitlines()
}

selectedMode = 'doodle'
model = models['doodle']
modelclass = classes['doodle']

fps = 300
screenWidth  = 1920
screenHeight = 1080

brushSize = 30
penColor  = [0, 0, 0]

buttonWidth, buttonHeight = 120, 35
labelWidth, labelHeight   = 400, 100

renderObjects = []

windll.shcore.SetProcessDpiAwareness(True)

pygame.init()

fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((screenWidth, screenHeight), pygame.FULLSCREEN)

font      = pygame.font.SysFont('Arial', 20)
largeFont = pygame.font.SysFont('Arial', 50)

canvasSize = [700, 700]
canvas = pygame.Surface(canvasSize)
canvas.fill((255, 255, 255))

class Button():
    def __init__(self, x, y, width, height, buttonText='Button', onclickFunction=None, onePress=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.onePress = onePress

        self.fillColors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }

        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))

        self.alreadyPressed = False

        renderObjects.append(self)

    def process(self):

        mousePos = pygame.mouse.get_pos()

        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])

            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])

                if self.onePress:
                    self.onclickFunction()

                elif not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True

            else:
                self.alreadyPressed = False

        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)

class Label():
    def __init__(self, x, y, width, height, labelText='Button'):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

        self.buttonSurf = largeFont.render(labelText, True, (20, 20, 20))

        renderObjects.append(self)

    def process(self):
        self.buttonSurface.fill("#ffffff")
        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)
    
    def setLabel(self, labelText):
        self.buttonSurf = largeFont.render(labelText, True, (20, 20, 20))

def changeColor(color):
    global penColor
    penColor = color

def clearScreen():
    canvas.fill((255,255,255))
    for Label in Labels:
        Label.setLabel('')

def sendPredict():
    pygame.image.save(canvas, "images/canvas.png")
    canvasImage = cv2.imread('images/canvas.png', cv2.IMREAD_GRAYSCALE)
    canvasImage = cv2.resize(canvasImage, (28, 28))
    canvasImage = np.array(canvasImage)
    canvasImage = canvasImage.reshape(28, 28, 1).astype('float32')
    canvasImage /= 255.0
    canvasImage = 1 - canvasImage
    predictions = predict_image(model, modelclass, canvasImage)
    index = 0
    for text in predictions:
        formattedText = text[0] + ' - ' + str(round(text[1], 2)) + '%'
        Labels[index].setLabel(formattedText)
        index+=1

def changeMode():
    global selectedMode, model, modelclass
    if selectedMode == 'doodle':
        selectedMode = 'alphabet'
        model = models['alphabet']
        modelclass = classes['alphabet']
        modelLabel.setLabel('ALPHABET MODE')
    else:
        selectedMode = 'doodle'
        model = models['doodle']
        modelclass = classes['doodle']
        modelLabel.setLabel('DOODLE MODE')

buttons = [
    ['Black', lambda: changeColor([0, 0, 0])],
    ['Eraser', lambda: changeColor([255, 255, 255])],
    ['Clear', lambda: clearScreen()],
    ['Predict', lambda: sendPredict()],
    ['Change Mode', lambda: changeMode()],
    ['Quit', quit],
]

labels = [ '', '', '', '', '' ]
Labels = []

modelLabel = Label(screenWidth - 500, labelWidth - 100 , labelWidth, labelHeight, "DOODLE MODE")

for index, buttonName in enumerate(buttons):
    Button(index * (buttonWidth + 10) + 10, 10, buttonWidth, buttonHeight, buttonName[0], buttonName[1])

for index, labelName in enumerate(labels):
    Labels.append(Label(screenWidth - 500, (index * 100) + labelWidth, labelWidth, labelHeight, labelName))

while True:
    screen.fill((30, 30, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    for object in renderObjects:
        object.process()

    x, y = screen.get_size()
    screen.blit(canvas, [x/2 - canvasSize[0]/2, y/2 - canvasSize[1]/2])

    if pygame.mouse.get_pressed()[0]:
        mx, my = pygame.mouse.get_pos()

        dx = mx - x/2 + canvasSize[0]/2
        dy = my - y/2 + canvasSize[1]/2

        pygame.draw.circle( canvas, penColor, [dx, dy], brushSize)

    pygame.draw.circle( screen, penColor, [100, 100], brushSize)

    pygame.display.flip()
    fpsClock.tick(fps)
