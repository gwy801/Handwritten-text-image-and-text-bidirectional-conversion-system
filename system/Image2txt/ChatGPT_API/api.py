from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def create_client():
    """创建 OpenAI 客户端"""
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL', 'https://api.chatanywhere.tech/v1')

    if not api_key:
        raise ValueError("API key not found in environment variables")

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def gpt_35_api(messages: list):
    client = create_client()
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in API call: {e}")
        return None


def gpt_35_api_stream(messages: list):
    client = create_client()
    try:
        stream = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            stream=True,
        )
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response
    except Exception as e:
        print(f"Error in stream API call: {e}")
        return None


def process_file(file_path, output_path='corrected_output.txt'):
    """读取整个文件内容并与 GPT 对话进行纠错"""
    try:
        # 读取整个文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        print(f"原始文本:\n{content}")

        # 设置对话消息
        messages = [
            {'role': 'system', 'content': 'Perform basic text modifications, only correcting incorrect words'},
            {'role': 'user', 'content': content}
        ]

        # 获取 GPT 回复
        corrected_text = gpt_35_api(messages)

        if corrected_text is None:
            print("Error: Failed to get response from API")
            return

        # 打印纠正后的文本
        print(f"\n纠正后的文本:\n{corrected_text}")

        # 将纠正后的文本写入输出文件
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(corrected_text)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    print("\n --- END ---")


def main():
    file_path = '../TrOCR/infer/recognition_results.txt'
    process_file(file_path)


if __name__ == "__main__":
    main()