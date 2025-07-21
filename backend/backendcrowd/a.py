from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize Flask app and YOLO model
app = Flask(__name__)
model = YOLO("best.pt")

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def crowd_analysis():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform prediction
    results = model.predict(source=file_path, conf=0.4)

    # Count detections for the "person" class
    person_class_id = 0  # COCO dataset assigns 0 to the 'person' class
    person_count = sum(1 for box in results[0].boxes if box.cls == person_class_id)

    # Render result template
    return render_template("result.html", image_url=url_for('uploaded_file', filename=file.filename), person_count=person_count)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return open(os.path.join(UPLOAD_FOLDER, filename), "rb").read(), 200, {
        "Content-Type": "image/jpeg"
    }

if __name__ == "__main__":
    app.run(debug=True, port=5001)
