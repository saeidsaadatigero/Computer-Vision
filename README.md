# :star::star::star::star::star:Image Classification with TensorFlow Hub:

This code demonstrates how to classify images using a pre-trained model from TensorFlow Hub, specifically the Inception V3 model. The code is designed to run in Google Colab and utilizes OpenCV for image processing.

## Requirements

To run this code, ensure you have the following libraries installed:

- OpenCV
- NumPy
- TensorFlow
- TensorFlow Hub
- JSON (comes with Python)

## Code Explanation

```python
from google.colab import files
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json  # Importing the JSON module
```

### Imports

- **files**: Allows file uploads in Google Colab.
- **cv2**: OpenCV library for image processing.
- **numpy**: Library for numerical operations.
- **tensorflow**: For building and executing machine learning models.
- **tensorflow_hub**: For loading pre-trained models from TensorFlow Hub.
- **json**: For handling JSON data.

---

```python
# Load the image classification model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/imagenet/inception_v3/classification/5")
```

### Model Loading

This line loads the Inception V3 model, which has been pre-trained on the ImageNet dataset. This model can classify images into one of 1000 categories.

---

```python
image_path = '/content/content.jpg'  # Update the filename as necessary
```

### Image Path

Specify the path to the image you want to classify. The default is set to `/content/content.jpg`, which is the typical upload location in Google Colab.

---

```python
# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (299, 299))  # Resize to 299x299 for InceptionV3
    img = img / 255.0  # Normalize the image
    img = img.astype(np.float32)  # Convert data type to float32
    return np.expand_dims(img, axis=0)  # Add batch dimension
```

### Image Preprocessing

This function performs several preprocessing steps:

1. **Read the Image**: Loads the image from the specified path.
2. **Error Handling**: Raises an error if the image cannot be loaded.
3. **Color Conversion**: Converts the image from BGR (OpenCV format) to RGB (standard format).
4. **Resize**: Resizes the image to 299x299 pixels, which is required by the Inception V3 model.
5. **Normalization**: Scales pixel values to the range [0, 1].
6. **Batch Dimension**: Adds an extra dimension to the image array to represent the batch size.

---

```python
# Function to load class names
def load_class_names():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    class_names = {}
    response = tf.keras.utils.get_file('imagenet_class_index.json', url)
    with open(response, 'r') as f:
        class_names = json.load(f)
    return {int(key): value[1] for key, value in class_names.items()}
```

### Class Names Loading

This function fetches and loads the class names corresponding to the ImageNet dataset:

1. **Fetch JSON**: Downloads a JSON file containing class indices and names.
2. **Load JSON**: Parses the JSON file to a Python dictionary.
3. **Return Class Names**: Returns a dictionary mapping class indices to class names.

---

```python
# Function to predict the image class
def predict_image(image_path, class_names):
    img = load_and_preprocess_image(image_path)  # Preprocess the image

    predictions = model(img)  # Make predictions
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the predicted class index
    predicted_probability = np.max(predictions)  # Get the highest probability

    return predicted_class, predicted_probability
```

### Image Prediction

This function predicts the class of the input image:

1. **Preprocessing**: Calls the preprocessing function.
2. **Prediction**: Uses the model to predict the class of the image.
3. **Class Index**: Retrieves the index of the class with the highest predicted probability.
4. **Probability**: Retrieves the maximum predicted probability.

---

```python
# Using the code
try:
    class_names = load_class_names()  # Load class names
    predicted_class, predicted_probability = predict_image(image_path, class_names)
    
    print(f'Predicted class: {class_names[predicted_class]}')
    print(f'Predicted probability: {predicted_probability:.4f}')
except ValueError as e:
    print(e)
```

### Execution

1. **Load Class Names**: Calls the function to load class names.
2. **Predict Image**: Calls the prediction function.
3. **Display Results**: Prints the predicted class name and its probability.
4. **Error Handling**: Catches and prints any errors that occur during execution.

---

## Conclusion

This code provides a straightforward method to classify images using a pre-trained model. By following the steps above, you can easily modify the image path and classify any image of your choice.

If you have any questions or need further assistance, feel free to open an issue!

--- 

Feel free to adjust any sections as needed for clarity or additional detail!
