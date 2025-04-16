# src/modules/glpm.py

import json
import logging
import os
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Please set your OpenAI API key in the environment variable OPENAI_API_KEY.")
else:
    api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

def generate_layout(parsed_json: dict) -> dict:
    """
    根据HPP解析结果生成初步布局信息。
    输出格式为包含每个对象和归一化bounding box的JSON，bbox格式为 [x_min, y_min, x_max, y_max]（取值范围0到1）。
    背景应始终输出为全图 [0, 0, 1, 1]。
    """
    system_msg = (
        "You are a layout planner for a diffusion model. "
        "Given a JSON tree representing a scene with objects and spatial relations, "
        "generate a JSON object that contains a key 'layout' with a list of items. "
        "Each item should be an object with two keys: 'object' and 'bbox'. "
        "The 'object' key is the name of the object and the 'bbox' is a list of four normalized floats representing [x_min, y_min, x_max, y_max]. "
        "The layout should cover all objects in 'objects_description' and include the background (which should always have bbox [0, 0, 1, 1]). "
        "Based on spatial relations (e.g., 'cat in center, dogs on left and right'), assign plausible bounding boxes. "
        "Do not include any extra explanation; output only valid JSON."
    )
    
    user_msg = (
        f"Based on the following parsed scene JSON from the HPP module, generate an initial layout with bounding boxes:\n\n"
        f"{json.dumps(parsed_json)}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=500
        )
        logging.info("Received response from GPT-4 API for layout planning.")
        layout_str = response.choices[0].message.content
        layout_json = json.loads(layout_str)
        return layout_json

    except Exception as e:
        logging.error(f"Error in generate_layout: {e}")
        raise e

if __name__ == "__main__":
    # 导入HPP模块，调用其parse_prompt函数
    try:
        # 确保 hpp.py 与当前文件在同一目录或者已在Python导入路径中
        from hpp import parse_prompt
    except ImportError as ie:
        logging.error("Failed to import parse_prompt from hpp.py. 请检查 hpp.py 模块位置。")
        raise ie

    demo_prompt = "A white cat and two yellow dogs and three birds playing on the grass."
    try:
        # 先解析Prompt获取语义树状结构
        parsed_json = parse_prompt(demo_prompt)
        logging.info("HPP module returned parsed JSON structure.")
        # 再根据解析结果生成布局信息
        layout = generate_layout(parsed_json)
        print("Generated layout JSON:")
        print(json.dumps(layout, indent=2))
    except Exception as error:
        print(f"Error during layout generation: {error}")