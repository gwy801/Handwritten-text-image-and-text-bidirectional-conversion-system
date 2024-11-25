import cv2
import pytesseract
import os
import shutil
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
from tqdm import tqdm


class TextSegmenter:
    def __init__(self, tesseract_path=r'C:\Program Files\Tesseract-OCR\tesseract.exe', max_workers=4):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.max_workers = max_workers
        self.thread_local = threading.local()
        self._lock = threading.Lock()

    @lru_cache(maxsize=32)
    def preprocess_image(self, image_array_bytes):
        """优化的图像预处理函数，使用缓存机制"""
        # 将字节转换回numpy数组
        image_array = np.frombuffer(image_array_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用CLAHE改善对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 快速的二值化处理
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 使用更小的kernel进行形态学操作
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return processed

    def clear_output_dir(self, output_dir):
        """清理输出目录"""
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def process_text_region(self, region_data):
        """处理单个文本区域"""
        x, y, w, h, text, image, padding, output_dir, counter = region_data

        min_x = max(0, x - padding)
        min_y = max(0, y - padding)
        max_x = min(x + w + padding, image.shape[1])
        max_y = min(y + h + padding, image.shape[0])

        line_image = image[min_y:max_y, min_x:max_x]

        with self._lock:
            line_filename = os.path.join(output_dir, f"image_{counter}.png")
            cv2.imwrite(line_filename, line_image)

        return (min_x, min_y, max_x, max_y)

    def segment_text_lines(self, image_path, output_dir, padding=10):
        """使用多线程优化的文本行分割"""
        self.clear_output_dir(output_dir)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Unable to read the image")

        original_image = image.copy()

        # 将图像转换为字节以用于缓存
        _, image_bytes = cv2.imencode('.png', image)
        processed = self.preprocess_image(image_bytes.tobytes())

        # 使用优化的Tesseract配置
        custom_config = r'--psm 6 --oem 3 -c tessedit_do_invert=0'
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=custom_config)

        # 收集有效的文本框
        valid_boxes = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                valid_boxes.append((x, y, w, h, text))

        if not valid_boxes:
            return

        # 使用DBSCAN进行聚类
        y_positions = np.array([box[1] for box in valid_boxes]).reshape(-1, 1)
        db = DBSCAN(eps=15, min_samples=1).fit(y_positions)

        # 准备多线程处理的数据
        process_data = []
        clustered_boxes = {}
        for i, label in enumerate(db.labels_):
            if label not in clustered_boxes:
                clustered_boxes[label] = []
            clustered_boxes[label].append(valid_boxes[i])

        for counter, (label, boxes) in enumerate(clustered_boxes.items()):
            min_x = min(box[0] for box in boxes)
            max_x = max(box[0] + box[2] for box in boxes)
            min_y = min(box[1] for box in boxes)
            max_y = max(box[1] + box[3] for box in boxes)

            process_data.append((
                min_x, min_y, max_x - min_x, max_y - min_y,
                "", image, padding, output_dir, counter
            ))

        # 使用线程池处理图像分割
        rectangles = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(process_data), desc="Processing text regions") as pbar:
                future_to_data = {executor.submit(self.process_text_region, data): data
                                  for data in process_data}

                for future in future_to_data:
                    try:
                        rectangle = future.result()
                        rectangles.append(rectangle)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing region: {str(e)}")

        # 在原图上标记识别区域
        for rect in rectangles:
            min_x, min_y, max_x, max_y = rect
            cv2.rectangle(original_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # 保存标记后的原图
        marked_image_path = os.path.join(output_dir, 'marked_image.png')
        cv2.imwrite(marked_image_path, original_image)

    def print_results(self, output_dir):
        """打印结果文件列表"""
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"Total files saved: {len(files)}")
        for filename in sorted(files):
            print(f"Saved: {filename}")


def main():
    # 设置处理器核心数
    num_cores = os.cpu_count()
    max_workers = max(1, num_cores - 2)  # 留出一个核心给系统使用

    segmenter = TextSegmenter(max_workers=max_workers)
    image_path = 'Data/img_1.png'
    output_dir = 'segmentation_output'

    try:
        segmenter.segment_text_lines(image_path, output_dir, padding=10)
        segmenter.print_results(output_dir)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()