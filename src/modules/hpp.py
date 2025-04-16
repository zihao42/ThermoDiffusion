# src/modules/hpp.py

import json
import logging
import os
from openai import OpenAI

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 检查 API Key 是否设置（建议通过环境变量设置）
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OpenAI API key in the environment variable OPENAI_API_KEY.")
else:
    api_key = os.environ["OPENAI_API_KEY"]

# 初始化 OpenAI 客户端
client = OpenAI(api_key=api_key)

def parse_prompt(prompt: str) -> dict:
    """
    接收用户输入的自由文本 Prompt（英语），通过调用 GPT-4 API 将其解析为树状结构，
    并转换成结构化的 JSON 数据返回。

    解析后的 JSON 结构示例：
    {
      "scene": {
        "layout_description": {
          "objects_count": {"cat": 1, "dog": 2},
          "spatial_relation": "cat in center, dogs on left and right"
        },
        "objects_description": {
          "cat": {"color": "white", "size": "small", "action": "playing"},
          "dog": {"color": "yellow", "size": "small", "action": "playing", "quantity": 2}
        },
        "background": "grass"
      }
    }

    :param prompt: 英文提示文本
    :return: 解析后的 JSON 数据（Python 字典）
    :raises Exception: 当 API 调用或 JSON 解析出错时抛出异常
    """
    # 构造系统提示，要求 GPT-4 返回严格 JSON 格式的树状结构数据
    system_msg = (
        "You are a prompt parser for a diffusion model. "
        "Given a free text prompt that describes a scene with objects, spatial relationships, and a background, "
        "produce a JSON tree structure with exactly the following keys: "
        "'scene' which contains 'layout_description', 'objects_description', and 'background'. "
        "Your response must be valid JSON without any extra explanations or markdown formatting. "
        "For example, if the prompt is 'A white cat and two yellow dogs playing on the grass', "
        "output something like: "
        '{"scene": {"layout_description": {"objects_count": {"cat": 1, "dog": 2}, "spatial_relation": "cat in center, dogs on left and right"}, '
        '"objects_description": {"cat": {"color": "white", "size": "small", "action": "playing"}, '
        '"dog": {"color": "yellow", "size": "small", "action": "playing", "quantity": 2}}, '
        '"background": "grass"}}'
    )
    user_msg = f"Parse the following prompt into a tree-structured JSON: '{prompt}'"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=500
        )
        parsed_content = response.choices[0].message.content
        logging.info("Received response from GPT-4 API.")
        parsed_json = json.loads(parsed_content)
        return parsed_json

    except Exception as e:
        logging.error(f"Error in parse_prompt: {e}")
        raise e


if __name__ == "__main__":
    demo_prompt = "A white cat and two yellow dogs playing on the grass."
    try:
        parsed_result = parse_prompt(demo_prompt)
        print("Parsed JSON structure:")
        print(json.dumps(parsed_result, indent=2))
    except Exception as error:
        print(f"Error during prompt parsing: {error}")
