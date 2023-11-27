import os
import xml.etree.ElementTree as ET

# 指定XML文件所在的目录列表
xml_directories = ['data1_WSODD/Annotations/', 'data2_leo/Annotations/', 'data3_lei/Annotations/']

# 用于存储所有不同的name字段值的集合
unique_names = set()
# 用于存储每个类别的数量
class_counts = {}

# 遍历每个目录
for xml_directory in xml_directories:
    # 遍历目录中的每个XML文件
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            # 构造完整的文件路径
            xml_path = os.path.join(xml_directory, filename)

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取所有<object>元素
            object_elements = root.findall('.//object')

            # 提取每个<object>元素中的<name>字段值并加入集合，并统计数量
            for object_element in object_elements:
                name_element = object_element.find('name')
                if name_element is not None:
                    unique_names.add(name_element.text)
                    class_counts[name_element.text] = class_counts.get(name_element.text, 0) + 1

# 打印所有不同的name字段值
print("Total number of unique object names:", len(unique_names))
print("Unique object names:", unique_names)

# 打印每个类别的数量
print("\nClass Counts:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
