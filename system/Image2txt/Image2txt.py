from Image2txt.Extract_singal_data.Extract import TextSegmenter
from Image2txt.TrOCR.infer.infer import ImageTextRecognizer
from Image2txt.ChatGPT_API.api import process_file
import os
import time  # 导入时间模块


def process_image_to_text(input_path):
    """将图片处理为文本的完整流程，并记录各步骤耗时"""
    try:
        # 初始化总计时
        total_start_time = time.time()

        # 步骤1：分割图片为单行文本
        step1_start_time = time.time()
        segmenter = TextSegmenter()  # 创建 TextSegmenter 类实例
        output_dir = 'segmentation_output'  # 输出文件夹路径
        segmenter.segment_text_lines(input_path, output_dir)  # 分割图像
        step1_end_time = time.time()
        print(f"步骤1: 图片分割耗时 {step1_end_time - step1_start_time:.2f} 秒")

        # 步骤2：识别图片中的文本
        step2_start_time = time.time()
        model_path = 'D:\桌面\图片分割与识别\Image2txt\TrOCR\TrOCR_Model'  # 模型文件夹路径
        recognizer = ImageTextRecognizer(model_path=model_path, image_folder=output_dir,output_txt_path='recognition_results.txt')  # 创建 ImageTextRecognizer 类实例
        recognizer.process_images()  # 处理并识别图片中的文本
        step2_end_time = time.time()
        print(f"步骤2: 文本识别耗时 {step2_end_time - step2_start_time:.2f} 秒")

        # 步骤3：纠错识别文本
        step3_start_time = time.time()
        recognition_txt_path = 'recognition_results.txt'  # 识别结果的文本文件路径
        output_path = 'corrected_output.txt'
        process_file(recognition_txt_path)  # 使用 process_file 进行文本纠错
        step3_end_time = time.time()
        print(f"步骤3: 文本纠错耗时 {step3_end_time - step3_start_time:.2f} 秒")

        # 返回纠错后的文本
        with open(output_path, 'r', encoding='utf-8') as file:
            total_end_time = time.time()
            print(f"总处理时间: {total_end_time - total_start_time:.2f} 秒")
            return file.read()

    except Exception as e:
        print(f"处理图片时出错: {e}")
        raise
