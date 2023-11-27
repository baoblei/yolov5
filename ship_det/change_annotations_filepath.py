import xml.etree.ElementTree as ET
import os


# 指定XML文件所在的目录
xml_directory = 'Annotations/'

# 遍历目录中的每个XML文件
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        # 构造完整的文件路径
        xml_path = os.path.join(xml_directory, filename)

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取<path>元素并修改其文本内容
        path_element = root.find('path')
        if path_element is not None:
            xml_name = os.path.basename(path_element.text) 
            path_element.text = 'Annotations/' + xml_name

        # 保存修改后的XML文件
        # modified_xml_path = os.path.join(xml_directory, 'modified_' + filename)
        tree.write(xml_path)

print("Done!")
