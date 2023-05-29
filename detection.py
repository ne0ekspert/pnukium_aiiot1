from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import time
import serial

ser = serial.Serial(port='COM8', baudrate=115200)
                    
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

prev_class = ''

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    print(image.shape)
    image = image[300:580,200:380]

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make the image a numpy array and reshape it to the models input shape.
    image_pred = np.asarray(image_pred, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_pred = (image_pred / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_pred)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    confidence_percent = str(np.round(confidence_score * 100))[:-2]+"%"

    cv2.putText(image, f"{class_name[2]} {confidence_percent}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    if prev_class == 'N' and class_name[2] != 'N':
        ser.write(class_name[2].encode('utf-8'))
        time.sleep(0.7)

    prev_class = class_name[2]

    # Print prediction and confidence score
    print("Class:", class_name[2], end="\t")
    print("Confidence Score:", confidence_percent)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
