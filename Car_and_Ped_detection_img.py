import cv2

# trained car data files
car_classifier_file = cv2.CascadeClassifier('cars_detection.xml')

# trained ped detection
ped_classifier_file = cv2.CascadeClassifier('peds_detection.xml')

img_source = 'car.jpg'

img = cv2.imread(img_source)

# converts img to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detects cars
car_coordinates = car_classifier_file.detectMultiScale(gray_img)

# detects pedestrians
ped_coordinates = ped_classifier_file.detectMultiScale(gray_img)


# creates the border around the cars
for (x, y, w, h) in car_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

for (x, y, w, h) in ped_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

# opens image
cv2.imshow('Car Detector', img)

# makes program wait for key press to close
cv2.waitKey()

print('Ran Successfully')
