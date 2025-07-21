from flask import Flask, request, render_template
import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import gdown
import matplotlib.pyplot as plt

# ------------------ Setup ------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive model URL (converted properly)
model_drive_url = "https://drive.google.com/uc?id=1GjDUWzrnpVsiCW9xEXg8AV3CxRgdGxxN"

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5...")
    gdown.download(model_drive_url, MODEL_PATH, quiet=False, fuzzy=True)

# ------------------ Custom Layer ------------------
class DistanceLayer(Layer):
    def call(self, anchor, compare):
        sum_squared = tf.reduce_sum(tf.square(anchor - compare), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# ------------------ Load Model ------------------
# model = load_model(MODEL_PATH, custom_objects={"DistanceLayer": DistanceLayer})
model = tf.keras.models.load_model(r"models\model.h5", custom_objects={"DistanceLayer": DistanceLayer})

# ------------------ Preprocess Function ------------------
def preprocess_image(image_path, img_size=(64, 64)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, axis=-1)  # (H, W, 1)
    image = np.expand_dims(image, axis=0)   # (1, H, W, 1)
    return image.astype(np.float32) / 255.0

# ------------------ Compare Function ------------------
def compare_images(img1, img2):
    return model.predict([img1, img2])[0][0]

# ------------------ Visualization ------------------
def visualize_result(image1_path, image2_path, similarity_score, threshold=0.3, output_path="static/output_comparison.png"):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        raise ValueError("Could not read one or both images.")

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    label = "Same Person" if similarity_score > threshold else "Different Person"

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis("off")
    plt.title("IMAGE 1")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis("off")
    plt.title("IMAGE 2")

    plt.suptitle(f"Result: {label} (Score: {similarity_score:.2f})", fontsize=16)
    plt.savefig(output_path)
    plt.close()

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Both images must be uploaded.", 400

        image1 = request.files['image1']
        image2 = request.files['image2']

        img1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
        img2_path = os.path.join(UPLOAD_FOLDER, image2.filename)

        image1.save(img1_path)
        image2.save(img2_path)

        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)

        score = compare_images(img1, img2)

        output_path = os.path.join('static', 'output_comparison.png')
        visualize_result(img1_path, img2_path, score, output_path=output_path)

        result_label = "MATCHED! THE IMAGES ARE OF THE SAME PERSON" if score >= 0.3 else "NOT MATCHED! THE IMAGES ARE NOT OF THE SAME PERSON"

        return render_template(
            'result.html',
            image1_url=f"/{img1_path}",
            image2_url=f"/{img2_path}",
            similarity_score=f"{score:.2f}",
            result_label=result_label,
            output_image_url=f"/{output_path}"
        )
    except Exception as e:
        return f"Error: {e}", 500

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True, port=5003)


# from flask import Flask, request, render_template
# import os
# import cv2
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Layer
# import matplotlib.pyplot as plt

# # Load the model with custom layer (DistanceLayer)
# class DistanceLayer(Layer):
#     def call(self, anchor, compare):
#         sum_squared = tf.reduce_sum(tf.square(anchor - compare), axis=1, keepdims=True)
#         return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# model = load_model('model.h5', custom_objects={"DistanceLayer": DistanceLayer})

# # Initialize Flask app
# app = Flask(__name__)

# # Directory to save uploaded images
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Preprocessing function for images
# def preprocess_image(image_path, img_size=(64, 64)):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale for the model
#     if image is None:
#         raise ValueError(f"Image not found: {image_path}")
#     image = cv2.resize(image, img_size)  # Resize the image to match the model's input size
#     image = np.expand_dims(image, axis=-1)  # Add the channel dimension (for grayscale images)
#     image = np.expand_dims(image, axis=0)  # Add the batch dimension
#     return image.astype(np.float32) / 255.0  # Normalize the image

# # Function to calculate similarity
# def compare_images(img1, img2):
#     score = model.predict([img1, img2])[0][0]  # Predict similarity score between the two images
#     return score

# # Visualization function
# def visualize_result(image1_path, image2_path, similarity_score, threshold=0.3, output_path="static/output_comparison.png"):
#     image1 = cv2.imread(image1_path)
#     image2 = cv2.imread(image2_path)
    
#     if image1 is None or image2 is None:
#         raise ValueError("Failed to load one or both images for visualization.")
    
#     # Convert BGR to RGB for display
#     image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#     image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

#     result_label = "Same Person" if similarity_score > threshold else "Different Person"

#     # Plot the images side by side
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image1)
#     plt.axis("off")
#     plt.title("IMAGE 1")

#     plt.subplot(1, 2, 2)
#     plt.imshow(image2)
#     plt.axis("off")
#     plt.title("IMAGE 2")

#     plt.suptitle(f"Result: {result_label} (Score: {similarity_score:.2f})", fontsize=16, color="black")

#     # Save the visualization to a file
#     plt.savefig(output_path)
#     plt.close()

# # Route to the index page (upload form)
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle image comparison
# @app.route('/compare', methods=['POST'])
# def compare_faces():
#     if 'image1' not in request.files or 'image2' not in request.files:
#         return "Both images must be uploaded.", 400

#     # Save the uploaded images
#     image1 = request.files['image1']
#     image2 = request.files['image2']

#     img1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
#     img2_path = os.path.join(UPLOAD_FOLDER, image2.filename)

#     image1.save(img1_path)
#     image2.save(img2_path)

#     # Preprocess the images
#     img1 = preprocess_image(img1_path)
#     img2 = preprocess_image(img2_path)

#     # Compare the images
#     score = compare_images(img1, img2)

#     # Create the result visualization
#     output_image_path = "static/output_comparison.png"
#     visualize_result(img1_path, img2_path, score, output_path=output_image_path)

#     # Render the result page
#     result_label = "MATCHED! THE IMAGES ARE OF THE SAME PERSON" if score >= 0.3 else "NOT MATCHED! THE IMAGES ARE NOT OF THE SAME PERSON"
#     return render_template('result.html', image1_url=f"/{img1_path}", image2_url=f"/{img2_path}", similarity_score=f"{score:.2f}", result_label=result_label, output_image_url=f"/{output_image_path}")

# if __name__ == '__main__':
#     app.run(debug=True, port=5003)


