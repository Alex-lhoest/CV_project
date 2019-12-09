import pandas as pd
import cv2 as cv

x1 = []
y1 = []
x2 = []
y2 = []

image_path = '../images/sudoku_51.png'
ground_path = '../annotations/sudoku_51.txt'

data = pd.read_csv(ground_path, header=None)
for _, row in data.iterrows():
    x1.append(int(row[0]))
    y1.append(int(row[1]))
    x2.append(int(row[2]))
    y2.append(int(row[3]))

img = cv.imread(image_path)

h, w = img.shape[:2]

for i in range(len(x1)):

    img = cv.line(img, (x1[i], h-y1[i]), (x2[i], h-y2[i]), (0, 255, 0))

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
