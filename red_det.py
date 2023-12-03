import cv2
import numpy as np
import os

def get_rect_area(rect):
    # 计算矩形框的面积
    return rect[2] * rect[3]

def detect_objects_infrared(image_path, temperature_threshold, distance, overlap, output_path=None):
    '''
    temperature_threshold: 温度阈值
    distance: 高亮度区域合并距离
    overlap: 重叠检测框合并
    '''
    # 读取红外图像
    if isinstance(image_path, str) and os.path.isfile(image_path):
        # 如果是文件路径，读取红外图像
        infrared_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # 如果不是文件路径，直接使用传入的图像
        infrared_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY) if image_path.shape[-1] == 3 else image_path

    # 上方边缘的mask
    infrared_image[:40] = 0

    # 根据温度阈值进行二值化
    _, binary_image = cv2.threshold(infrared_image, temperature_threshold, 255, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 生成矩形框表示
    rects=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rects.append([x,y,w,h])

    rects = merge_rects(rects, distance, overlap)

    # 保存图像
    if output_path is not None:
        result_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)
        for rect in rects:
            x,y,w,h = rect
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(output_path,result_image)
    
    return rects


def merge_rects(rects, min_distance, min_overlap):
    def calculate_overlap(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        overlap_area = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2

        overlap_ratio1 = overlap_area / area1
        overlap_ratio2 = overlap_area / area2

        return max(overlap_ratio1, overlap_ratio2)

    def distance(rect1, rect2):
        x1, y1, _, _ = rect1
        x2, y2, _, _ = rect2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def merge_two_rects(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        return [x, y, w, h]

    merged_rects = []
    visited = set()

    for i in range(len(rects)):
        if i in visited:
            continue

        current_rect = rects[i]
        merged_rect = current_rect.copy()

        for j in range(i + 1, len(rects)):
            if j in visited:
                continue

            other_rect = rects[j]
            dist = distance(merged_rect, other_rect)
            overlap = calculate_overlap(merged_rect, other_rect)

            if overlap>0:
                overlap

            if dist < min_distance or overlap > min_overlap:
                merged_rect = merge_two_rects(merged_rect, other_rect)
                visited.add(j)

        merged_rects.append(merged_rect)

    return merged_rects


def merge_close_contours(contours, min_distance_threshold):
    # 合并离得近的轮廓
    merged_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        added = False

        for i, merged_contour in enumerate(merged_contours):
            x_merged, y_merged, w_merged, h_merged = cv2.boundingRect(merged_contour)

            # 如果当前轮廓完全包含在已合并的轮廓中，则保留
            if x >= x_merged and y >= y_merged and x + w <= x_merged + w_merged and y + h <= y_merged + h_merged:
                # merged_contours[i] = np.vstack([merged_contour, contour])
                added = True
                break
            if x < x_merged and y < y_merged and x + w > x_merged + w_merged and y + h > y_merged + h_merged:
                merged_contours[i] = contour
                added = True
                break
            # 如果两个包围框之间的距离小于阈值，则合并
            if abs(x - x_merged) < min_distance_threshold and abs(y - y_merged) < min_distance_threshold:
                merged_contours[i] = np.vstack([merged_contour, contour])
                added = True
                break

        if not added:
            merged_contours.append(contour)

    return merged_contours 


if __name__=='__main__':
    # 设置温度阈值和合并距离阈值
    temperature_threshold = 200
    distance= 300
    overlap = 0.8

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
            detect_objects_infrared(input_path, temperature_threshold, distance, overlap, output_path)

