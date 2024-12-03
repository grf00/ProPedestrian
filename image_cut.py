# 读取图像
import cv2
import os


def crop_and_save_images(input_path, output_dir, prefix='cropped_image', start_index=1):
    """
    判断输入的文件格式（图像或视频），并执行相应的截图操作。
    如果是视频，取第一帧进行截图。
    文件名格式：cropped_image_输入路径文件名_第i个截图.jpg

    :param input_path: 输入的图像或视频路径
    :param output_dir: 截图保存的文件夹路径
    :param prefix: 截图文件名前缀，默认为 'cropped_image'
    :param start_index: 截图编号的起始索引，默认为 1
    """
    # 判断输入文件是图像还是视频
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))  # 根据文件扩展名判断是否是视频

    # 提取输入路径的文件名（不带路径和扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    if is_video:
        # 如果是视频，读取视频的第一帧
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧，操作失败。")
            return
        img = frame  # 获取视频的第一帧
        cap.release()
        print("已从视频中提取第一帧进行截图。")
    else:
        # 如果是图像，直接读取图像文件
        img = cv2.imread(input_path)
        if img is None:
            print("无法读取图像文件，操作失败。")
            return

    index = start_index
    while True:
        # 使用cv2.selectROI选择截图区域
        roi = cv2.selectROI("Select Region (Press 'q' to quit)", img)

        # 如果选定了区域（roi不为None）
        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            cropped_img = img[y:y+h, x:x+w]

            # 如果指定的文件夹不存在，则创建它
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 自动生成文件名并保存截图
            output_filename = f"{prefix}_{input_filename}_{index}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"截图 {output_filename} 已保存到: {output_path}")

            # 增加索引，准备保存下一个截图
            index += 1
        else:
            print("未选择区域，操作已取消。")
            break

    # 关闭图像窗口
    cv2.destroyAllWindows()
vedio_path ='example/input-vedio/行人检测视频01.flv'
image_path = 'example/input-image/reid-2/frame246.jpg'  # 替换为你的图片路径
output_folder='yolov7/fast_reid_master/datasets/query'
crop_and_save_images(vedio_path, output_folder)