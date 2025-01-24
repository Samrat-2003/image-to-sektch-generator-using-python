from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import os
from werkzeug.utils import secure_filename
import zipfile
from io import BytesIO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image, style='default'):
    # Convert uploaded file to numpy array
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        nparr = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image.seek(0)  # Reset file pointer for reuse
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sketches = {}
    
    # Default sketch
    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    sketches['default'] = cv2.divide(gray, blur, scale=256.0)
    
    # Detailed sketch
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sketches['detailed'] = cv2.divide(gray, blur, scale=256.0)
    
    # Pencil sketch
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    sketch = cv2.divide(gray, blur, scale=256.0)
    sketches['pencil'] = cv2.multiply(sketch, 0.8)  # Adjust brightness
    
    # Convert to base64
    processed_sketches = {}
    for style_name, sketch in sketches.items():
        _, buffer = cv2.imencode('.jpg', sketch)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        processed_sketches[style_name] = base64_image
    
    return processed_sketches

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    processed_images = []
    
    for file in files:
        if file and allowed_file(file.filename):
            # Process the image in all styles
            sketches = process_image(file)
            processed_images.append(sketches)
    
    return jsonify({'sketches': processed_images})

@app.route('/download-all', methods=['POST'])
def download_all():
    try:
        # Create a ZIP file in memory
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for file_idx, sketches in enumerate(request.json.get('sketches', [])):
                for style, image_data in sketches.items():
                    # Convert base64 to bytes
                    img_data = base64.b64decode(image_data)
                    # Add to zip with organized folder structure
                    zf.writestr(f'sketches/image_{file_idx + 1}/{style}_sketch.jpg', img_data)
        
        # Prepare the response
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='all_sketches.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 