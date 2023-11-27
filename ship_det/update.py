import os
import xml.etree.ElementTree as ET

# 指定XML文件所在的目录列表
xml_directories = ['data2_leo/Annotations/', 'data3_lei/Annotations/']

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

            # 修改<object>元素中<name>字段为'people'的值为'person'
            for object_element in object_elements:
                name_element = object_element.find('name')
                if name_element is not None and name_element.text == 'ship':
                    name_element.text = 'boat'

            # 保存修改后的XML文件
            tree.write(xml_path)

print("Done!")
