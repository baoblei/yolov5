# coding:utf-8

import os
import random
import argparse
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default=['data1_WSODD/Annotations/', 'data2_leo/Annotations/', 'data3_lei/Annotations/', 'data4/Annotations/'], type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0  # 训练集和验证集所占比例。 这里没有划分测试集
train_percent = 0.9     # 训练集所占比例，可自己进行调整
xmlfilepaths = opt.xml_path
txtsavepath = opt.txt_path

# 合并所有XML文件
total_xml = []

# 遍历每个目录
for xml_directory in xmlfilepaths:
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

            # 判断文件是否包含 'person'、'ship'、'boat' 类别
            contains_categories = any(
                object_element.find('name').text in ['person', 'ship', 'boat']
                for object_element in object_elements
            )

            # 如果包含目标类别，则将文件名加入列表
            if contains_categories:
                total_xml.append(filename)


# for xmlfilepath in xmlfilepaths:

#     total_xml.extend(os.listdir(xmlfilepath))

# 统计包含 'person'、'ship'、'boat' 类别的XML文件数量
num_selected_files = len(total_xml)
print(f"Total number of XML files containing 'person', 'ship', or 'boat': {num_selected_files}")
# print("Selected XML files:", total_xml)

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
