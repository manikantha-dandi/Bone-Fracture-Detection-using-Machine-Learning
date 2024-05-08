from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import joblib  # For loading scikit-learn models
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__, template_folder='C:/Users/DELL/Desktop/fracture Project')
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load your model
model = joblib.load('random_forest_model_nand.pkl')  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = process_and_predict(file_path)
            return jsonify({'prediction': prediction})
    return render_template('upload.html')

def process_and_predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 512))  # Resize as your model expects
    edges = cv2.Canny(img, 100, 150)  # Apply Canny edge detection
    features = extract_features(edges)  # Use edges for feature extraction
    prediction = model.predict([features])
    return int(prediction[0])

class GLCM:
    def __init__(self, image):
        self.distances = [1, 3, 5, 8]
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
        self.glcm_mat = graycomatrix(image, distances=self.distances, angles=self.angles, symmetric=True, normed=True)
        self.properties = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast']

    def extract_features(self):
        return np.hstack([graycoprops(self.glcm_mat, prop).ravel() for prop in self.properties])

def extract_features(image):
    glcm = GLCM(image)
    return glcm.extract_features()

if __name__ == '__main__':
    app.run(debug=True)
