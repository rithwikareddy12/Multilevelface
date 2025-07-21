# from flask import Flask, request, render_template, redirect, url_for
# import os
# import cv2
# import numpy as np
# from mtcnn import MTCNN
# from keras_facenet import FaceNet
# import pickle
# from pymongo import MongoClient

# app = Flask(__name__)

# # Initialize MTCNN and FaceNet
# detector = MTCNN()
# embedder = FaceNet()

# # MongoDB connection
# client = MongoClient("mongodb://localhost:27017/")
# db = client["faces"]  # Replace with your database name
# collection = db["users"]  # Replace with your collection name

# # Paths
# encodings_file = r"C:\Users\Rithwika reddy\Desktop\finalproject\project\server\multiface1.pkl"
# output_image_path = "static/output.jpg"

# # Load encodings
# with open(encodings_file, "rb") as f:
#     data = pickle.load(f)
#     known_encodings = np.array(data["encodings"])
#     known_names = data["names"]

# def preprocess_face(face):
#     """Resize and normalize face image for model prediction."""
#     face = cv2.resize(face, (160, 160))
#     face = face.astype("float32") / 255.0
#     face = np.expand_dims(face, axis=0)
#     return face

# def recognize_faces(image_path, known_encodings, known_names, threshold=0.5):
#     """Detect and recognize faces in an image."""
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     faces = detector.detect_faces(rgb_image)
#     recognized_roll_numbers = []  # Store all recognized roll numbers

#     for face in faces:
#         x, y, width, height = face["box"]
#         x, y = max(0, x), max(0, y)

#         face_roi = rgb_image[y:y+height, x:x+width]
#         preprocessed_face = preprocess_face(face_roi)
#         embedding = embedder.model.predict(preprocessed_face)[0]

#         distances = np.linalg.norm(known_encodings - embedding, axis=1)
#         min_distance = np.min(distances)

#         if min_distance < threshold:
#             recognized_roll_numbers.append(known_names[np.argmin(distances)])
#         else:
#             recognized_roll_numbers.append("Unknown")

#         # Draw face box and name on the image
#         cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
#         cv2.putText(image, recognized_roll_numbers[-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#     # Save the output image
#     cv2.imwrite(output_image_path, image)
#     return recognized_roll_numbers

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return "No file uploaded", 400

#     file = request.files['image']
#     file_path = os.path.join("uploaded_image.jpg")
#     file.save(file_path)

#     # Recognize face(s) and get roll numbers
#     recognized_roll_numbers = recognize_faces(file_path, known_encodings, known_names, threshold=0.7)

#     # Join all recognized roll numbers into a string (comma-separated for multiple faces)
#     roll_numbers = ', '.join(recognized_roll_numbers)

#     # Redirect to result page with the roll number(s)
#     return redirect(url_for('result', roll_numbers=roll_numbers))

# @app.route('/result')
# def result():
#     # Get the roll numbers from the URL query parameter (it could be multiple, separated by commas)
#     roll_numbers = request.args.get('roll_numbers', 'Unknown')

#     # Assuming the processed image is saved as 'output.jpg' in the 'static' folder
#     image_url = url_for('static', filename='output.jpg')

#     return render_template('result.html', image_url=image_url, roll_numbers=roll_numbers)


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle
from pymongo import MongoClient

app = Flask(__name__)

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["faces"]  # Replace with your database name
collection = db["imagedocuments"]  # Replace with your collection name

# Paths
encodings_file = r"C:\Users\Rithwika reddy\Desktop\Sem2_1files\finalproject\project1\project\server\multiface1.pkl"
output_image_path = "static/output.jpg"

# Load encodings
with open(encodings_file, "rb") as f:
    data = pickle.load(f)
    known_encodings = np.array(data["encodings"])
    known_names = data["names"]

def preprocess_face(face):
    """Resize and normalize face image for model prediction."""
    face = cv2.resize(face, (160, 160))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def retrieve_person_data(rollno):
    """Retrieve the data associated with the recognized person from MongoDB."""
    person_data = collection.find_one({"qrData.rollNo": rollno})  # Adjust query for nested field
    if person_data:
        qr_data = person_data.get("qrData", {})
        return {
            "rollNo": qr_data.get("rollNo", "N/A"),
            "name": qr_data.get("name", "N/A"),
            "fatherName": qr_data.get("fatherName", "N/A"),
            "department": qr_data.get("department", "N/A"),
            "contact": qr_data.get("contact", "N/A"),
            "images_count": len(person_data.get("images", [])),
        }
    else:
        return {"rollNo": rollno, "name": "Unknown"}

def recognize_faces(image_path, known_encodings, known_names, threshold=0.5):
    """Detect and recognize faces in an image."""
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb_image)
    recognized_details = []  # Store details of recognized persons

    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)

        face_roi = rgb_image[y:y+height, x:x+width]
        preprocessed_face = preprocess_face(face_roi)
        embedding = embedder.model.predict(preprocessed_face)[0]

        distances = np.linalg.norm(known_encodings - embedding, axis=1)
        min_distance = np.min(distances)

        if min_distance < threshold:
            rollno = known_names[np.argmin(distances)]
            person_data = retrieve_person_data(rollno)
            recognized_details.append(person_data)
        else:
            recognized_details.append({"rollNo": "Unknown", "name": "Unknown"})

        # Draw face box and roll number on the image
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, rollno if min_distance < threshold else "Unknown", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Save the output image
    cv2.imwrite(output_image_path, image)
    return recognized_details

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    file_path = os.path.join("uploaded_image.jpg")
    file.save(file_path)

    # Recognize face(s) and get details
    recognized_details = recognize_faces(file_path, known_encodings, known_names, threshold=0.7)

    # Redirect to result page with the details
    return render_template('result.html', image_url=url_for('static', filename='output.jpg'), recognized_details=recognized_details)

if __name__ == "__main__":
    app.run(debug=True)
