import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ocr import HindiTextRecognizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        recognizer = HindiTextRecognizer()

        try:
            recognized_text = recognizer.visualize_and_save_results(filepath)
        except Exception as e:
            app.logger.error(f'Error during OCR processing: {e}')
            return jsonify({'error': 'Failed to process image'}), 500

        return jsonify({"text": recognized_text})

    app.logger.error('Invalid file type')
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True)