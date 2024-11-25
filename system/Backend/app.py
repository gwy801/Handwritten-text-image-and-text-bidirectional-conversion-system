import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from Text2pic.HandWrite import generate_handwriting
from Image2txt.Image2txt import process_image_to_text

app = Flask(__name__, static_folder="output")  # 设置静态文件夹
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image part", 400
    image = request.files['image']
    if image.filename == '':
        return "No selected file", 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(img_path)

    text = process_image_to_text(img_path)
    return text


@app.route('/generate-handwriting', methods=['POST'])
def generate_handwriting_route():
    data = request.get_json()
    text = data.get('text', '')
    params = data.get('params', {})
    params['font_file']=data.get('font')
    if not text:
        return jsonify({'error': 'Text input is required'}), 400

    try:
        images = generate_handwriting(text, params)

        # 返回图片路径（URL 格式）
        image_urls = [f"/output/{os.path.basename(img)}" for img in images]

        return jsonify(image_urls)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 路由，用于返回静态文件（图像）
@app.route('/output/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
