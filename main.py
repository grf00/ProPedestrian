import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QAction, QFileDialog, QMessageBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from yolo_fastreid_lama import yolo_fastreid_lama
from yolov7.yolo_fastreid_detect import fast_detect
from anony_recory import recover_images
sys.path.append(".")
sys.path.append("lama/configs")
sys.path.append("lama/models")
sys.path.append("reidstrong")
sys.path.append("yolov7/fast_reid_master")

class PrivacyProtectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("隐私保护应用")
        self.setGeometry(100, 100, 1000, 600)  # 设置窗口大小

        self.initUI()

    def initUI(self):
        # 创建菜单栏
        menubar = self.menuBar()

        # 创建四个独立的菜单
        screenshot_menu = menubar.addMenu('截图')
        detection_menu = menubar.addMenu('检测')
        encryption_menu = menubar.addMenu('加密')
        decryption_menu = menubar.addMenu('解密')

        # 为截图菜单添加动作
        screenshot_action = QAction('截图功能', self)
        screenshot_action.triggered.connect(self.show_screenshot_page)
        screenshot_menu.addAction(screenshot_action)

        # 为检测菜单添加动作
        detection_action = QAction('检测功能', self)
        detection_action.triggered.connect(self.show_detection_page)
        detection_menu.addAction(detection_action)

        # 为加密菜单添加动作
        encryption_action = QAction('加密功能', self)
        encryption_action.triggered.connect(self.show_encryption_page)
        encryption_menu.addAction(encryption_action)

        # 为解密菜单添加动作
        decryption_action = QAction('解密功能', self)
        decryption_action.triggered.connect(self.show_decryption_page)
        decryption_menu.addAction(decryption_action)

        # 创建堆栈式小部件管理页面
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 初始化各个页面
        self.screenshot_page = QWidget()
        self.detection_page = QWidget()
        self.encryption_page = QWidget()
        self.decryption_page = QWidget()

        # 填充页面内容
        self.init_screenshot_page()
        self.init_detection_page()
        self.init_encryption_page()
        self.init_decryption_page()

        # 将页面添加到堆栈
        self.stacked_widget.addWidget(self.screenshot_page)
        self.stacked_widget.addWidget(self.detection_page)
        self.stacked_widget.addWidget(self.encryption_page)
        self.stacked_widget.addWidget(self.decryption_page)

        # 默认显示截图页面
        self.stacked_widget.setCurrentWidget(self.screenshot_page)

    def init_screenshot_page(self):
        layout = QVBoxLayout()

        # 创建标签显示图片
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedSize(800, 600)

        # 创建一个水平布局来包裹 image_label，使其居中
        image_layout = QHBoxLayout()
        image_layout.addStretch(1)
        image_layout.addWidget(self.image_label)
        image_layout.addStretch(1)

        # 创建按钮
        self.open_button = QPushButton("选择图片", self)
        self.open_button.clicked.connect(self.open_file)

        self.crop_button = QPushButton("手动截图", self)
        self.crop_button.clicked.connect(self.enable_crop_mode)
        self.crop_button.setEnabled(False)  # Initially disabled

        # 创建按钮的布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.crop_button)

        # 将水平布局添加到垂直布局中
        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        self.screenshot_page.setLayout(layout)

        # Internal variables
        self.current_image = None
        self.output_dir = os.path.join(os.getcwd(), "yolov7/fast_reid_master/datasets/query")

    def init_detection_page(self):
        layout = QVBoxLayout()

        # 创建两个QLabel用于显示图像
        self.detection_original_label = QLabel(self)
        self.detection_original_label.setAlignment(Qt.AlignCenter)
        self.detection_original_label.setScaledContents(True)
        self.detection_original_label.setStyleSheet("border: 1px solid black;")
        self.detection_original_label.setFixedSize(400, 300)

        self.detection_result_label = QLabel(self)
        self.detection_result_label.setAlignment(Qt.AlignCenter)
        self.detection_result_label.setScaledContents(True)
        self.detection_result_label.setStyleSheet("border: 1px solid black;")
        self.detection_result_label.setFixedSize(400, 300)

        # 创建按钮
        self.load_image_button = QPushButton("加载图片", self)
        self.load_image_button.clicked.connect(self.load_image_for_detection)

        self.load_folder_button = QPushButton("加载文件夹", self)
        self.load_folder_button.clicked.connect(self.load_folder_for_detection)

        self.detect_button = QPushButton("检测", self)
        self.detect_button.clicked.connect(self.run_detection)

        # 布局：将两个QLabel放在同一行
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.detection_original_label)
        image_layout.addWidget(self.detection_result_label)

        # 布局：将按钮放在一行
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.load_folder_button)
        button_layout.addWidget(self.detect_button)

        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        self.detection_page.setLayout(layout)

    def init_encryption_page(self):
        layout = QVBoxLayout()

        # 创建两个QLabel用于显示图像
        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setScaledContents(True)
        self.original_image_label.setStyleSheet("border: 1px solid black;")
        self.original_image_label.setFixedSize(400, 300)

        self.encrypted_image_label = QLabel(self)
        self.encrypted_image_label.setAlignment(Qt.AlignCenter)
        self.encrypted_image_label.setScaledContents(True)
        self.encrypted_image_label.setStyleSheet("border: 1px solid black;")
        self.encrypted_image_label.setFixedSize(400, 300)

        # 创建按钮
        self.load_image_button = QPushButton("加载图片", self)
        self.load_image_button.clicked.connect(self.load_image_for_encryption)

        self.load_folder_button = QPushButton("加载文件夹", self)
        self.load_folder_button.clicked.connect(self.load_folder_for_encryption)

        self.encrypt_button = QPushButton("加密", self)
        self.encrypt_button.clicked.connect(self.encrypt_image)

        # 布局：将两个QLabel放在同一行
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.encrypted_image_label)

        # 布局：将按钮放在一行
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.load_folder_button)
        button_layout.addWidget(self.encrypt_button)

        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        self.encryption_page.setLayout(layout)

    def init_decryption_page(self):
        layout = QVBoxLayout()

        # 创建两个QLabel用于显示图像
        self.decryption_original_label = QLabel(self)
        self.decryption_original_label.setAlignment(Qt.AlignCenter)
        self.decryption_original_label.setScaledContents(True)
        self.decryption_original_label.setStyleSheet("border: 1px solid black;")
        self.decryption_original_label.setFixedSize(400, 300)

        self.decrypted_image_label = QLabel(self)
        self.decrypted_image_label.setAlignment(Qt.AlignCenter)
        self.decrypted_image_label.setScaledContents(True)
        self.decrypted_image_label.setStyleSheet("border: 1px solid black;")
        self.decrypted_image_label.setFixedSize(400, 300)

        # 创建按钮
        self.load_image_button_decryption = QPushButton("加载图片", self)
        self.load_image_button_decryption.clicked.connect(self.load_image_for_decryption)

        self.load_folder_button_decryption = QPushButton("加载文件夹", self)
        self.load_folder_button_decryption.clicked.connect(self.load_folder_for_decryption)

        self.decrypt_button = QPushButton("解密", self)
        self.decrypt_button.clicked.connect(self.decrypt_image)

        # 布局：将两个QLabel放在同一行
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.decryption_original_label)
        image_layout.addWidget(self.decrypted_image_label)

        # 布局：将按钮放在一行
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button_decryption)
        button_layout.addWidget(self.load_folder_button_decryption)
        button_layout.addWidget(self.decrypt_button)

        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        self.decryption_page.setLayout(layout)

    def show_screenshot_page(self):
        self.stacked_widget.setCurrentWidget(self.screenshot_page)

    def show_detection_page(self):
        self.stacked_widget.setCurrentWidget(self.detection_page)

    def show_encryption_page(self):
        self.stacked_widget.setCurrentWidget(self.encryption_page)

    def show_decryption_page(self):
        self.stacked_widget.setCurrentWidget(self.decryption_page)

    def load_image_for_detection(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./", "Images (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.load_image(file_path, self.detection_original_label)

    def load_folder_for_detection(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if folder_path:
            self.detection_folder_path = folder_path
            # 获取文件夹中的第一张图片
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
            if image_files:
                first_image_path = os.path.join(folder_path, image_files[0])
                self.load_image(first_image_path, self.detection_original_label)
                print(f"已选择文件夹：{folder_path}")
            else:
                print("文件夹中没有有效的图像文件。")
        else:
            print("未选择文件夹。")

    def run_detection(self):
        try:
            if hasattr(self, 'detection_folder_path'):
                if not os.path.exists(self.detection_folder_path):
                    QMessageBox.warning(self, "无效路径", "所选文件夹路径无效。")
                    return

                # 调用检测功能
                self.detection_result_folder = "./runs/detect3"
                os.makedirs(self.detection_result_folder, exist_ok=True)
                fast_detect(self.detection_folder_path,self.detection_result_folder)

                # 显示检测后的第一张图像
                detection_result_files = [f for f in os.listdir(self.detection_result_folder+'/exp') if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
                if detection_result_files:
                    first_result_image_path = os.path.join(self.detection_result_folder+'/exp', detection_result_files[0])
                    result_image = cv2.imread(first_result_image_path)
                    self.display_image(result_image, self.detection_result_label)
                    QMessageBox.information(self, "检测完成", "检测成功！")
                else:
                    QMessageBox.warning(self, "检测失败", "检测后没有生成图像。")
            else:
                QMessageBox.warning(self, "未加载文件夹", "请先加载文件夹。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测过程中发生错误: {str(e)}")
            print(e)
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./", "Images (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.load_image(file_path, self.image_label)
    def load_image(self, file_path, target_label):
        self.current_image_path = file_path
        self.current_image = cv2.imread(file_path)
        if self.current_image is None:
            target_label.setText("无法加载图像，请检查文件格式或内容。")
            return
        self.display_image(self.current_image, target_label)
        if hasattr(self, 'crop_button'):
            self.crop_button.setEnabled(True)  # 仅截图界面需要此功能

    def display_image(self, image, target_label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        qimage = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        target_label.setPixmap(pixmap)
    def enable_crop_mode(self):
        if self.current_image is not None:
            # Initialize variables for saving cropped images
            index = 1  # Starting index for file naming
            output_dir = self.output_dir  # Directory to save the cropped images
            prefix = "0001_c1s1_0_0"  # Prefix for filenames

            while True:
                # Open an interactive ROI selection window
                roi = cv2.selectROI("Select Region (Press 'Enter' to confirm, 'Esc' to quit)", self.current_image)

                if roi != (0, 0, 0, 0):
                    x, y, w, h = roi
                    cropped_img = self.current_image[y:y + h, x:x + w]

                    # Ensure the output directory exists
                    os.makedirs(output_dir, exist_ok=True)

                    # Generate the filename and save the cropped image
                    output_filename = f"{prefix}{index}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Screenshot {output_filename} saved to: {output_path}")

                    # Increment index for the next screenshot
                    index += 1
                else:
                    print("No region selected, exiting crop mode.")
                    break

            # Close the ROI selection window
            cv2.destroyAllWindows()
        else:
            print("No image loaded. Please load an image before cropping.")

    def load_image_for_encryption(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./", "Images (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.load_image(file_path, self.original_image_label)

    def load_folder_for_encryption(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if folder_path:
            self.folder_path = folder_path
            # 获取文件夹中的第一张图片
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
            if image_files:
                first_image_path = os.path.join(folder_path, image_files[0])
                self.load_image(first_image_path, self.original_image_label)
                print(f"已选择文件夹：{folder_path}")
            else:
                print("文件夹中没有有效的图像文件。")
        else:
            print("未选择文件夹。")

    def encrypt_image(self):
        try:
            if hasattr(self, 'folder_path'):
                if not os.path.exists(self.folder_path):
                    QMessageBox.warning(self, "无效路径", "所选文件夹路径无效。")
                    return
                # 加密文件夹中的所有图像
                self.encryption_folder = "example/lama_s_pic"
                os.makedirs(self.encryption_folder, exist_ok=True)
                yolo_fastreid_lama(self.folder_path,1)

                # 加密后的第一张图像
                encrypted_image_files = [f for f in os.listdir(self.encryption_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
                if encrypted_image_files:
                    first_encrypted_image_path = os.path.join(self.encryption_folder, encrypted_image_files[0])
                    encrypted_image = cv2.imread(first_encrypted_image_path)
                    self.display_image(encrypted_image, self.encrypted_image_label)
                    QMessageBox.information(self, "加密完成", "加密成功！")
                else:
                    QMessageBox.warning(self, "加密失败", "加密后没有生成图像。")
            else:
                QMessageBox.warning(self, "未加载文件夹", "请先加载文件夹。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加密过程中发生错误: {str(e)}")

    def load_image_for_decryption(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "./", "Images (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.load_image(file_path, self.decryption_original_label)

    def load_folder_for_decryption(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "./")
        if folder_path:
            self.decryption_folder_path = folder_path
            # 获取文件夹中的第一张图片
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
            if image_files:
                first_image_path = os.path.join(folder_path, image_files[0])
                self.load_image(first_image_path, self.decryption_original_label)
                print(f"已选择文件夹：{folder_path}")
            else:
                print("文件夹中没有有效的图像文件。")
        else:
            print("未选择文件夹。")

    def decrypt_image(self):
        try:
            if hasattr(self, 'decryption_folder_path'):
                if not os.path.exists(self.decryption_folder_path):
                    QMessageBox.warning(self, "无效路径", "所选文件夹路径无效。")
                    return
                # 调用解密功能
                self.decryption_result_folder = "example/cover_sec_rec"
                os.makedirs(self.decryption_result_folder, exist_ok=True)
                # 假设anony_recovery是解密功能模块
                processor = recover_images(fps=25,im_dir=self.decryption_folder_path)
                processor.process()
                # 显示解密后的第一张图像
                decrypted_image_files = [f for f in os.listdir(self.decryption_result_folder) if
                                         f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
                if decrypted_image_files:
                    first_decrypted_image_path = os.path.join(self.decryption_result_folder, decrypted_image_files[0])
                    decrypted_image = cv2.imread(first_decrypted_image_path)
                    self.display_image(decrypted_image, self.decrypted_image_label)
                    QMessageBox.information(self, "解密完成", "解密成功！")
                else:
                    QMessageBox.warning(self, "解密失败", "解密后没有生成图像。")
            else:
                QMessageBox.warning(self, "未加载文件夹", "请先加载文件夹。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"解密过程中发生错误: {str(e)}")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PrivacyProtectionApp()
    window.show()
    sys.exit(app.exec_())
