import sys
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
import shutil

# Ensure the default encoding is UTF-8 for console output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

# Path to the encodings file
encodings_file = "multiface1.pkl"

def preprocess_face(face):
    """Resize and normalize face image for model prediction."""
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def load_existing_encodings(encodings_file):
    """Load existing encodings from the file if it exists."""
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"]
    return [], []

def save_encodings(encodings_file, data):
    """Safely save encodings to the file."""
    temp_file = encodings_file + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(data, f)
    shutil.move(temp_file, encodings_file)
def generate_encodings(dataset_path, encodings_file):
    """Generate average encodings for each person and update the file."""
    # Load existing encodings and names
    existing_encodings, existing_names = load_existing_encodings(encodings_file)

    # Prepare to store new encodings and names
    new_encodings = []
    new_names = []

    if not os.path.isdir(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)

        # Ensure it's an image file
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}. Skipping...")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)

        if len(faces) == 0:
            print(f"No faces found in {image_path}. Skipping...")
            continue

        # Assume the first detected face is the correct one
        face = faces[0]
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        face_roi = rgb_image[y:y+height, x:x+width]

        # Preprocess face and generate embedding
        preprocessed_face = preprocess_face(face_roi)
        embedding = embedder.model.predict(preprocessed_face)[0]
        new_encodings.append(embedding)
        new_names.append(os.path.basename(dataset_path))  # Use folder name as the person's name

    # Merge new encodings with existing ones
    combined_encodings = existing_encodings + new_encodings
    combined_names = existing_names + new_names

    # Save updated encodings and names to file
    save_encodings(encodings_file, {"encodings": combined_encodings, "names": combined_names})
    print(f"Encodings updated and saved to {encodings_file}")

if __name__ == "__main__":
    # Get dataset path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python generate_encodings.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    generate_encodings(dataset_path, encodings_file)
