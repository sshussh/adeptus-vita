import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0,
)  # Pre-trained model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import img_to_array  # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.efficientnet import preprocess_input  # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from PIL import Image
import os
import pickle
import sys
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_and_preprocess_image(image_path, image_size):
    """
    Loads and preprocesses an image for EfficientNetB0.

    Parameters:
    - image_path (str): Path to the image file.
    - image_size (tuple): Target size to resize the image.

    Returns:
    - Preprocessed image as a NumPy array.
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(image_size)  # Resize to target dimensions
    img_array = img_to_array(img)  # Convert to array (height, width, 1)
    img_array = np.repeat(img_array, 3, axis=-1)  # Replicate across 3 channels
    img_array = preprocess_input(img_array)  # EfficientNet specific preprocessing
    return img_array


def PredictNewInstance(
    image_path, model, label_encoder, *, showImage=True, imageSize=(300, 300)
):
    """
    Predicts the class of a new image using a trained classifier and features extracted from EfficientNetB0.

    Parameters:
    - image_path (str): Path to the input image.
    - model (keras.Model or sklearn-like): Classifier model that takes image features as input.
    - showImage (bool): Whether to display the input image (default: True).
    - imageSize (tuple): Size to resize the image for EfficientNetB0 input (default: (300, 300)).

    Returns:
    - prediction (int): Predicted class.
    """

    # Load and preprocess the image
    img = load_and_preprocess_image(image_path, (300, 300))

    # Load EfficientNetB0 as a feature extractor (without the final classification layers)
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,  # Exclude final classification layer
        pooling="avg",  # Global average pooling
        input_shape=(imageSize[0], imageSize[1], 3),
    )
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

    # Create a batch with a single image
    img_batch = np.expand_dims(img, axis=0)

    # Extract CNN features
    features = feature_extractor.predict(img_batch)

    # Predict class probabilities using the given classifier
    prediction_probs = model.predict(features, verbose=0)

    # Get the class with the highest probability
    prediction = np.argmax(prediction_probs, axis=1)[0]

    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return predicted_label


model = load_model(os.path.join(os.getcwd(), "app", "api", "ANN.h5"))
with open(os.path.join(os.getcwd(), "app", "api", "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Predict dementia from MRI image")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )

    args = parser.parse_args()

    try:
        prediction = PredictNewInstance(args.image_path, model, label_encoder)
        print(prediction)
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
