import json
import re
import os

def convert_txt_to_json(txt_file, json_file):
    """
    将 txt 文件转换为 JSON 格式
    按照主要章节标题分割
    字段名从章节标题提取
    """
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 字段名映射规则
    field_mapping = {
        'Instructions:': 'instructions',
        'Game Strategy for Player 0:': 'policy',
        'Game Strategy for Player 1:': 'policy',
        'Legal actions:': 'actions',
        'Facts:': 'facts',
        'Outputs:': 'outputs',
        'Examples:': 'examples'
    }

    # 找到所有章节的起始位置
    section_patterns = list(field_mapping.keys())
    sections = []

    for pattern in section_patterns:
        pos = content.find(pattern)
        if pos != -1:
            sections.append((pos, pattern))

    # 按位置排序
    sections.sort(key=lambda x: x[0])

    result = {}

    # 提取每个章节的内容
    for i, (pos, pattern) in enumerate(sections):
        field_name = field_mapping[pattern]

        # 确定章节结束位置
        if i < len(sections) - 1:
            end_pos = sections[i + 1][0]
            section_content = content[pos:end_pos].strip()
        else:
            # 最后一个章节，取到文件末尾
            section_content = content[pos:].strip()

        result[field_name] = section_content

    # 保存为 JSON，使用美化的格式
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=0, ensure_ascii=False)

    print(f"✓ 转换完成: {txt_file} -> {json_file}")
    print(f"  字段: {list(result.keys())}")

def main():
    scenario = "cramped_room"
    txt_files = [
        f'{scenario}/{scenario}_0.txt',
        f'{scenario}/{scenario}_1.txt'
    ]

    for txt_file in txt_files:
        if os.path.exists(txt_file):
            json_file = txt_file.replace('.txt', '.json')
            convert_txt_to_json(txt_file, json_file)
        else:
            print(f"✗ 文件不存在: {txt_file}")

if __name__ == '__main__':
    main()
