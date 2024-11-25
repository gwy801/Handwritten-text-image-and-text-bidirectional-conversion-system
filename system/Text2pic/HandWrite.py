import time
from handright import Template, handwrite
from PIL import Image, ImageDraw, ImageFont
import os
import io
import shutil
import tempfile
import matplotlib.pyplot as plt
import re
from flask import jsonify
from datetime import datetime

# Get current path
current_path = os.getcwd()

# Create output directory if it doesn't exist
output_path = os.path.join(current_path, "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Preprocess the text for formatting and cleaning
def preprocess_txt(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(' ', '  ')  # Double spaces
    paragraphs = text.split('\n\n')
    formatted_text = "\n\n".join(paragraphs)
    return formatted_text

def create_notebook_image(width, height, line_spacing, top_margin, bottom_margin, left_margin, right_margin):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = top_margin
    lines = []
    while y < height - bottom_margin:
        main_line = (left_margin, y, width - right_margin, y)
        lines.append((main_line, (0, 0, 0)))  # Black line

        aux_y = y + line_spacing // 2
        if aux_y < height - bottom_margin:
            lines.append(((left_margin, aux_y, width - right_margin, aux_y), (180, 180, 180)))  # Gray line

        y += line_spacing

    lines.sort(key=lambda line: sum(line[1]) / 3, reverse=True)

    for line, color in lines:
        draw.line(line, fill=color, width=1)

    return image

def generate_handwriting(text, params):
    text = preprocess_txt(text)

    background_image = create_notebook_image(
        width=int(params.get('width', 1800)),
        height=int(params.get('height', 2500)),
        line_spacing=int(params.get('line_spacing', 100)),
        top_margin=int(params.get('top_margin', 150)),
        bottom_margin=int(params.get('bottom_margin', 150)),
        left_margin=int(params.get('left_margin', 150)),
        right_margin=int(params.get('right_margin', 150))
    )
    print(params)

    font_path = os.path.join("../Text2pic/font_assets", params.get('font_file', '李国夫手写体.ttf'))

    if not os.path.exists(font_path):
        print(f"Font file not found at {font_path}")
        return {"error": "Font file not found"}, 400

    font = ImageFont.truetype(font_path, size=int(params.get('font_size', 80)))

    template = Template(
        background=background_image,
        font=font,
        line_spacing=int(params.get('line_spacing', 100)),
        left_margin=int(params.get('left_margin', 150)),
        top_margin=int(params.get('top_margin', 150)),
        right_margin=int(params.get('right_margin', 150)),
        bottom_margin=int(params.get('bottom_margin', 150)),
        word_spacing=int(params.get('word_spacing', -10)),
        line_spacing_sigma=int(params.get('line_spacing_sigma', 2)),
        font_size_sigma=int(params.get('font_size_sigma', 1)),
        word_spacing_sigma=int(params.get('word_spacing_sigma', 1)),
        end_chars="，。！？；：",
        perturb_x_sigma=float(params.get('perturb_x_sigma', 1)),
        perturb_y_sigma=float(params.get('perturb_y_sigma', 1)),
        perturb_theta_sigma=float(params.get('perturb_theta_sigma', 0.02))
    )

    images = handwrite(text, template)
    if not images:
        print("No images generated!")
        return {"error": "No images generated"}, 500

    image_paths=[]
    for i, im in enumerate(images):
        # 获取当前时间戳并格式化为字符串
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(output_path, f"{timestamp}_{i}.png")

        im.save(image_path, format='PNG')
        image_paths.append(image_path)

        print(f"Image {i} saved to {image_path}")


    return image_paths  # 只返回本地路径
