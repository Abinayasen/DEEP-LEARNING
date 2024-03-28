from keras.models import load_model
import cv2
import numpy as np

img_size=150
DIRECTORY = "D:\AI\DEEP LEARNING\DATA\DATA_FLOWER"
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# Function to preprocess the input image
def preprocess_image(img_path):
    img = cv2.imread(img_path)  # Read the image using OpenCV
    img = cv2.resize(img, (img_size, img_size))  # Resize the image to match the input size of your model
    img = img / 255.0  # Normalize the pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class of the input image
def predict_flower_class(img_path):
    # Load the trained model
    model = load_model(r"D:\AI\DEEP LEARNING\flower_classification.keras")

    # Preprocess the input image
    img = preprocess_image(img_path)

    # Predict the class of the input image
    prediction = model.predict(img)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Map the predicted class index to the flower category
    predicted_flower = CATEGORIES[predicted_class_index]

    return predicted_flower

# Path to the input image provided by the user

input_image_path = r"D:\AI\DEEP LEARNING\tulip.jpeg"  # Provide the path to the user's input image

# Predict the flower class of the input image
predicted_flower = predict_flower_class(input_image_path)
print("The predicted flower in the image is:", predicted_flower)
