import cv2
import numpy as np
import os

def detect_smoke_and_save(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历input文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅处理图像文件
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            image = cv2.imread(image_path)

            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 定义烟雾的颜色范围（可以根据实际情况调整）
            lower_smoke = np.array([0, 42, 0], dtype=np.uint8)
            upper_smoke = np.array([179, 255, 120], dtype=np.uint8)

            # 根据颜色范围创建掩码
            mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

            # 使用形态学操作进行去噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 判断是否有轮廓存在，存在则认为检测到烟雾
            if len(contours) > 0:
                print(f"烟雾检测：{filename} - 烟雾存在!")

                # 在图像上绘制轮廓
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

                # 保存带有轮廓的图像到output文件夹
                cv2.imwrite(output_path, image)
            else:
                print(f"烟雾检测：{filename} - 烟雾不存在.")

# 调用检测方法
detect_smoke_and_save('smoke_det/input/', 'smoke_det/output/')
