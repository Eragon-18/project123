import numpy as np 
import matplotlib.pyplot as plt
import cv2
import seaborn as sns 
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PIL import Image
import time
import os

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("data.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "c", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=2500, train_size = 7500, random_state=42)
xTrainScaled = xTrain/255
xTestScaled = xTest/255

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xTrainScaled, yTrain)
yPredict = clf.predict(xTestScaled)
accuracy = accuracy_score(yTest, yPredict)
print("Accuracy: ", accuracy)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomRight = (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]
        imagePil = Image.fromarray(roi)
        image_bw = imagePil.convert("L")
        image_resized = image_bw.resize((28, 28), Image.ANTIALIAS)


        imageinverted = PIL.ImageOps.invert(image_resized)
        pixel_filter = 20
        min_pixel = np.percentile(imageinverted, pixel_filter)
        imagescaled = np.clip(imageinverted - min_pixel, 0, 255)
        max_pixel = np.max(imageinverted)
        imagescaled = np.asarray(imagescaled)/max_pixel

        testsample = np.array(imagescaled).reshape(1, 784)
        testpredict = clf.predict(testsample)
        print("Prediction: ", testpredict)

        cv2.imshow("Frame", gray)
        if cv2.waitkey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
    