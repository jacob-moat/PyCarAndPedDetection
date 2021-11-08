import cv2

# trained car detection
car_classifier_file = cv2.CascadeClassifier('cars_detection.xml')

# trained ped detection
ped_classifier_file = cv2.CascadeClassifier('peds_detection.xml')

# video source
video_source = cv2.VideoCapture('carsandpeds.mp4')
# 0 is default video device on pc, change 0 to video file to see car detection on video

while True:
    successful_frame_read, frame = video_source.read()

    # converts each frame into grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects cars
    car_coordinates = car_classifier_file.detectMultiScale(grayscale_frame)

    # detects pedestrians
    ped_coordinates = ped_classifier_file.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in ped_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # creates the viewer
    cv2.imshow('Car And Ped Detector', frame)

    key = cv2.waitKey(1)

    # assigns q to close the program
    if key == 81 or key == 113:
        break

video_source.release()
