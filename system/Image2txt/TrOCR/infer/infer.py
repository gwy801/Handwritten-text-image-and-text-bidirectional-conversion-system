from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time  # 导入时间模块


class ImageTextRecognizer:
    def __init__(self, model_path, image_folder, output_txt_path='recognition_results.txt', max_workers=4):
        """初始化方法，加载模型和图片路径"""
        self.model_path = model_path
        self.image_folder = image_folder
        self.output_txt_path = output_txt_path
        self.max_workers = max_workers

        # 加载模型和处理器
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

        # 将模型移到 CPU
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def extract_number(self, filename):
        """提取文件名中的数字部分，用于排序"""
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    def process_single_image(self, image_file):
        """处理单张图片的方法"""
        try:
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path).convert("RGB")

            # 处理图像
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # 进行推理
            generated_ids = self.model.generate(pixel_values)

            # 解码结果
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #
            # # 输出识别信息
            # print(f"识别文件: {image_file}, 识别结果: {generated_text}")

            return (self.extract_number(image_file), generated_text)
        except Exception as e:
            print(f"处理图片 {image_file} 时发生错误: {str(e)}")
            return (self.extract_number(image_file), "")

    def process_images(self):
        """使用多线程批量处理文件夹中的所有图片"""
        # 获取所有图片文件
        image_files = [f for f in os.listdir(self.image_folder)
                       if f.endswith('.png') and f.startswith('image')]

        # 创建进度条
        pbar = tqdm(total=len(image_files), desc="Processing images")

        # 存储所有识别结果
        results = []

        # 使用线程池处理图片
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.process_single_image, image_file): image_file
                              for image_file in image_files}

            # 获取结果
            for future in future_to_file:
                try:
                    index, text = future.result()
                    results.append((index, text))
                    pbar.update(1)
                except Exception as e:
                    print(f"获取结果时发生错误: {str(e)}")

        pbar.close()

        # 按索引排序结果
        results.sort(key=lambda x: x[0])
        for _,text in  results:
            print(_,text)
        recognition_results = [text for _, text in results]

        # 保存结果
        self.save_recognition_results(recognition_results)

    def save_recognition_results(self, results):
        """将识别结果保存到txt文件，不保留原来的文本内容"""

        # 将新结果写入文件，直接覆盖原内容
        with open(self.output_txt_path, 'w', encoding='utf-8') as output_file:
            output_file.write(' '.join(results))

        print(f"\n所有识别结果已保存到 {self.output_txt_path}")


# 示例用法：
if __name__ == "__main__":
    # 设置模型路径和图片文件夹路径
    model_path = '../TrOCR_Model'
    image_folder = '../../Extract_singal_data/segmentation_output/'

    # 创建ImageTextRecognizer类实例，设置4个工作线程
    recognizer = ImageTextRecognizer(
        model_path=model_path,
        image_folder=image_folder,
        max_workers=6  # 可以根据CPU核心数调整
    )

    # 记录开始时间
    start_time = time.time()

    # 调用处理图片并保存结果的方法
    recognizer.process_images()

    # 记录结束时间
    end_time = time.time()

    # 计算并输出运行时间
    print(f"总运行时间: {end_time - start_time:.2f}秒")