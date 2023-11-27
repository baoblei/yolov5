import cv2
import numpy as np
import os

def detect_objects_infrared(image_path, temperature_threshold, min_distance_threshold, output_path):
    # 读取红外图像
    infrared_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 上方边缘的mask
    infrared_image[:40] = 0

    # 根据温度阈值进行二值化
    _, binary_image = cv2.threshold(infrared_image, temperature_threshold, 255, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 合并离得近的轮廓
    merged_contours = merge_close_contours(contours, min_distance_threshold)
    # 合并重叠的轮廓
    merged_contours = merge_overlapping_contours(merged_contours)

    result_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)
    # 在原始图像上绘制检测到的轮廓
    # cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2) 
    # 在原始图像上绘制合并后的方框
    for contour in merged_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 保存图像
    cv2.imwrite(output_path,result_image)


def merge_close_contours(contours, min_distance_threshold):
    # 合并离得近的轮廓
    merged_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        added = False

        for i, merged_contour in enumerate(merged_contours):
            x_merged, y_merged, w_merged, h_merged = cv2.boundingRect(merged_contour)

            # 如果两个包围框之间的距离小于阈值，则合并
            if abs(x - x_merged) < min_distance_threshold and abs(y - y_merged) < min_distance_threshold:
                merged_contours[i] = np.vstack([merged_contour, contour])
                added = True
                break

        if not added:
            merged_contours.append(contour)

    return merged_contours 

def merge_overlapping_contours(contours):
    # 合并重叠的轮廓
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    rectangles = []

    for box in bounding_boxes:
        x, y, w, h = box
        rectangles.append(((x, y), (x + w, y + h)))

    rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.2)
    merged_contours = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) for ((x, y), (w, h)) in rectangles]

    return merged_contours


# 设置温度阈值和合并距离阈值
temperature_threshold = 220
min_distance_threshold = 130

output_folder = 'red_img/output/'
input_folder = 'red_img/input/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 确保文件是图像文件
        # 构造完整的输入和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 处理单个图像
        detect_objects_infrared(input_path, temperature_threshold, min_distance_threshold, output_path)

