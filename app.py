import cv2
import numpy as np
import os
from flask import Flask, render_template

app = Flask(__name__)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return {}

    # Get image dimensions
    height, width, channels = image.shape
    dimensions = f"{width}x{height}"

    # Perform inverse transformation
    inverse_image = cv2.bitwise_not(image)
    inverse_path = "static/inverse_image.jpg"
    cv2.imwrite(inverse_path, inverse_image)

    # Contrast Stretching
    min_val, max_val, _, _ = cv2.minMaxLoc(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    contrast_stretch = cv2.convertScaleAbs(image, alpha=255/(max_val-min_val), beta=-min_val*255/(max_val-min_val))
    contrast_path = "static/contrast_image.jpg"
    cv2.imwrite(contrast_path, contrast_stretch)

    # Histogram Equalization
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray_image)
    hist_eq_path = "static/hist_eq_image.jpg"
    cv2.imwrite(hist_eq_path, hist_eq)

    # Edge Detection (Canny)
    edges = cv2.Canny(gray_image, 100, 200)
    edges_path = "static/edges_image.jpg"
    cv2.imwrite(edges_path, edges)

    return {
        "input": image_path,
        "inverse": inverse_path,
        "contrast": contrast_path,
        "hist_eq": hist_eq_path,
        "edges": edges_path,
    }, dimensions, channels

@app.route("/")
def display_images():
    image_path = "static/input_image.jpg"  # Place input image in the 'static/' folder
    images, dimensions, channels = process_image(image_path)
    return render_template("index.html", images=images, dimensions=dimensions, channels=channels)

if __name__ == "__main__":
    app.run(debug=True)
