import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageProcessor:
    @staticmethod
    def preprocess(img):
        """CLAHE增强与双边滤波"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(img_clahe, 9, 75, 75)

    @staticmethod
    def feature_extraction(orig_img, mask):
        """特征提取与可视化"""
        masked = cv2.bitwise_and(orig_img, orig_img, mask=mask)
        edges = cv2.Canny(masked, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = orig_img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        return result_img

    """图像处理功能类"""
    @staticmethod
    def to_gray(image):
        """灰度化"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def contrast_adjust(image, value):
        """对比度调整"""
        alpha = (value + 100) / 100
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    @staticmethod
    def nonoise(image, method="gaussian", kernel_size=3):
        """去噪"""
        if method == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "median":
            return cv2.medianBlur(image, kernel_size)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            return image

    @staticmethod
    def edge_jiance(image, method="canny", threshold1=50, threshold2=150):
        """边缘检测"""
        gray = ImageProcessor.to_gray(image) if len(image.shape) == 3 else image
        if method == "canny":
            return cv2.Canny(gray, threshold1, threshold2)
        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5,
                                   cv2.convertScaleAbs(sobely), 0.5, 0)
        elif method == "laplacian":
            return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        else:
            return gray

    @staticmethod
    def gray_stretch(image, method="linear", maxval=255, **kwargs):
        """灰度拉伸（修正版）"""
        # 确保输入是灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if method == "linear":
            return ImageProcessor.linear_stretch(gray)
        elif method == "logarithmic":
            c = kwargs.get('c', 255.0)
            return ImageProcessor.duishu_stretch(gray, c, maxval)
        elif method == "power-law":
            gamma = kwargs.get('gamma', 1.0)
            return ImageProcessor.power_law_stretch(gray, gamma, maxval)
        else:
            return gray

    @staticmethod
    def morphological_operations(image, operation="dilation", kernel_size=3):
        """形态学处理"""
        # 确保输入是灰度图
        gray = ImageProcessor.to_gray(image) if len(image.shape) == 3 else image

        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        if operation == "erosion":  # "腐蚀"
            return cv2.erode(gray, kernel)
        elif operation == "dilation":  # "膨胀"
            return cv2.dilate(gray, kernel)
        elif operation == "opening":  # "开运算"
            return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        elif operation == "closing":  # "闭运算"
            return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        elif operation == "gradient":  # "形态学梯度"
            return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        else:
            return gray

    @staticmethod
    def to_binary(image, thresh=127, maxval=255):
        """二值化阈值处理到图像"""
        # 确保图像是灰度图
        gray = ImageProcessor.to_gray(image) if len(image.shape) == 3 else image
        # 应用二值化阈值处理
        _, result = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        return result

    @staticmethod
    def linear_stretch(image):
        """线性拉伸（修正版）"""
        # 确保输入是numpy数组
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是numpy数组")

        # 处理单通道图像
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用OpenCV的normalize进行线性拉伸
        stretched = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return stretched.astype(np.uint8)

    @staticmethod
    def duishu_stretch(image, c=255.0, maxval=255):
        """对数拉伸（修正版）"""
        # 输入验证
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是numpy数组")

        # 转换为float32避免溢出
        img_float = image.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        # 防止除以零
        if max_val - min_val <= 0:
            return image.astype(np.uint8)

        # 对数拉伸公式
        stretched = maxval * (np.log(1 + (img_float - min_val)) / np.log(1 + c))

        # 转换回uint8
        return np.clip(stretched, 0, 255).astype(np.uint8)

    @staticmethod
    def xingtai_processing(image, method="erode", kernel_size=3, iterations=1):
        #形态学处理
        # 转换为灰度图(如果是彩色图像)
        gray = ImageProcessor.to_gray(image) if len(image.shape) == 3 else image

        # 创建方形结构元素(可改为参数控制形状)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        try:
            if method == "erode":#"erode"(腐蚀),
                return cv2.erode(gray, kernel, iterations=iterations)
            elif method == "dilate":  # "dilate"(膨胀),
                return cv2.dilate(gray, kernel, iterations=iterations)
            elif method == "open": # "open"(开运算),
                return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif method == "close":# "close"(闭运算),
                return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif method == "gradient":# "gradient"(形态学梯度)
                # 梯度运算忽略iterations参数
                return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            else:
                raise ValueError(f"不支持的形态学方法: {method}")

        except Exception as e:
            # 错误处理(保持与原有代码风格一致)
            print(f"形态学处理错误: {str(e)}")
            return gray

    @staticmethod
    def power_law_stretch(image, gamma=1.0, maxval=255):
        """幂律拉伸（修正版）"""
        # 输入验证
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是numpy数组")

        # 转换为float32
        img_float = image.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        # 防止除以零
        if max_val - min_val <= 0:
            return image.astype(np.uint8)

        # 归一化并应用幂律变换
        normalized = (img_float - min_val) / (max_val - min_val)
        stretched = maxval * np.power(normalized, gamma)

        # 转换回uint8
        return np.clip(stretched, 0, 255).astype(np.uint8)

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("盲道识别系统")
        self.root.geometry("1000x800")
        # 盲道色彩
        self.lower_hsv = np.array([15, 40, 120], dtype=np.uint8)  # 默认黄色下限
        self.upper_hsv = np.array([35, 255, 255], dtype=np.uint8)  # 默认黄色上限
        # 初始化变量
        self.mask = None
        self.original_image = None
        self.processed_image = None
        self.original_label = None
        self.processed_label = None
        self.plate_box = None
        self.plate_text = None
        self.photo_original = None
        self.photo_processed = None
        # 创建UI
        self.create_widgets()

    def create_widgets(self):
        # 创建样式对象并设置颜色方案
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', background='#4a7abc', foreground='black',
                        font=('Arial', 10, 'bold'), borderwidth=1)
        style.configure('TLabelframe', background='#e0e0e0', borderwidth=2)
        style.configure('TLabelframe.Label', background='#4a7abc', foreground='white')
        style.configure('TLabel', background='#e0e0e0', font=('Arial', 9))
        style.configure('TCombobox', font=('Arial', 9))
        style.map('TButton', background=[('active', '#5a8adc')])

        # 特殊按钮样式
        style.configure('Process.TButton', background='#4caf50', foreground='black')
        style.map('Process.TButton', background=[('active', '#66bb6a')])
        style.configure('Action.TButton', background='#ff9800', foreground='black')
        style.map('Action.TButton', background=[('active', '#ffac33')])

        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_panel = ttk.LabelFrame(main_frame, text="图像处理控制", width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)

        # 按钮区域分组
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # 图像操作按钮（使用蓝色）
        self.load_button = ttk.Button(button_frame, text="加载图像", command=self.load_image)
        self.load_button.pack(fill=tk.X, pady=3)

        # 特殊功能按钮（使用橙色）
        self.chuli_button = ttk.Button(button_frame, text="定位盲道", command=self.start_locate_thread,
                                       style='Action.TButton')
        self.chuli_button.pack(fill=tk.X, pady=3)
        self.fenge_button = ttk.Button(button_frame, text="盲道分割", command=self.show_segmentation,
                                       style='Action.TButton')
        self.fenge_button.pack(fill=tk.X, pady=3)

        self.save_button = ttk.Button(button_frame, text="保存结果", command=self.save_image)
        self.save_button.pack(fill=tk.X, pady=3)

        # 添加分隔线
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=10, padx=5)

        # 处理方法选择
        method_frame = ttk.Frame(control_panel)
        method_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        self.method_label = ttk.Label(method_frame, text="处理方法:")
        self.method_label.pack(side=tk.LEFT)

        self.method_combo = ttk.Combobox(method_frame, values=[
            "原始图像", "灰度化", "对比度调整", "去噪", "灰度拉伸",
            "边缘检测", "形态学处理", "二值化"], state="readonly", width=15)
        self.method_combo.current(0)
        self.method_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.method_combo.bind("<<ComboboxSelected>>", self.update_visible_params)

        # 参数控制区域
        self.param_frame = ttk.LabelFrame(control_panel, text="参数调整")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)

        # 滑块样式统一
        slider_style = ttk.Style()
        slider_style.configure('Horizontal.TScale', background='#e0e0e0')

        # 对比度调整滑块
        self.contrast_frame = ttk.Frame(self.param_frame)
        self.contrast_label = ttk.Label(self.contrast_frame, text="对比度: 100%")
        self.contrast_label.pack(side=tk.LEFT)
        self.contrast_slider = ttk.Scale(
            self.contrast_frame, from_=0, to=200, style='Horizontal.TScale',
            command=lambda e: self.contrast_label.config(text=f"对比度: {int(float(e))}%"))
        self.contrast_slider.set(100)
        self.contrast_slider.pack(fill=tk.X, expand=True, padx=5)

        # 去噪方法选择
        self.denoise_method_label = ttk.Label(self.param_frame, text="去噪方法:")
        self.denoise_method_combo = ttk.Combobox(
            self.param_frame, values=["高斯模糊", "中值滤波", "双边滤波"], state="readonly", width=12)
        self.denoise_method_combo.current(0)

        # 去噪核大小滑块
        self.denoise_kernel_frame = ttk.Frame(self.param_frame)
        self.denoise_kernel_label = ttk.Label(self.denoise_kernel_frame, text="核大小: 3")
        self.denoise_kernel_label.pack(side=tk.LEFT)
        self.denoise_kernel_slider = ttk.Scale(
            self.denoise_kernel_frame, from_=3, to=15, style='Horizontal.TScale',
            command=lambda e: self.denoise_kernel_label.config(text=f"核大小: {int(float(e))}"))
        self.denoise_kernel_slider.set(3)
        self.denoise_kernel_slider.pack(fill=tk.X, expand=True, padx=5)

        # 边缘检测方法选择
        self.edge_method_label = ttk.Label(self.param_frame, text="边缘检测方法:")
        self.edge_method_combo = ttk.Combobox(
            self.param_frame, values=["Canny", "Sobel", "Laplacian"], state="readonly", width=10)
        self.edge_method_combo.current(0)
        self.edge_method_combo.bind("<<ComboboxSelected>>", self.update_edge_params)

        # Canny阈值滑块
        self.canny_thresh1_frame = ttk.Frame(self.param_frame)
        self.canny_thresh1_label = ttk.Label(self.canny_thresh1_frame, text="阈值1: 50")
        self.canny_thresh1_label.pack(side=tk.LEFT)
        self.canny_thresh1_slider = ttk.Scale(
            self.canny_thresh1_frame, from_=0, to=255, style='Horizontal.TScale',
            command=lambda e: self.canny_thresh1_label.config(text=f"阈值1: {int(float(e))}"))
        self.canny_thresh1_slider.set(50)
        self.canny_thresh1_slider.pack(fill=tk.X, expand=True, padx=5)

        self.canny_thresh2_frame = ttk.Frame(self.param_frame)
        self.canny_thresh2_label = ttk.Label(self.canny_thresh2_frame, text="阈值2: 150")
        self.canny_thresh2_label.pack(side=tk.LEFT)
        self.canny_thresh2_slider = ttk.Scale(
            self.canny_thresh2_frame, from_=0, to=255, style='Horizontal.TScale',
            command=lambda e: self.canny_thresh2_label.config(text=f"阈值2: {int(float(e))}"))
        self.canny_thresh2_slider.set(150)
        self.canny_thresh2_slider.pack(fill=tk.X, expand=True, padx=5)

        # 灰度拉伸
        self.gray_stretch_method_label = ttk.Label(self.param_frame, text="灰度拉伸方法:")
        self.gray_stretch_method_combo = ttk.Combobox(
            self.param_frame, values=["线性拉伸", "对数拉伸", "幂律拉伸"], state="readonly", width=10)
        self.gray_stretch_method_combo.current(0)
        self.gray_stretch_method_combo.bind("<<ComboboxSelected>>", self.update_stretch_params)

        # 灰度拉伸参数滑块
        self.stretch_frame = ttk.Frame(self.param_frame)
        self.stretch_label = ttk.Label(self.stretch_frame, text="参数: 1.0")
        self.stretch_label.pack(side=tk.LEFT)
        self.stretch_slider = ttk.Scale(
            self.stretch_frame, from_=0.1, to=5.0, style='Horizontal.TScale',
            command=lambda e: self.stretch_label.config(text=f"参数: {float(e):.1f}"))
        self.stretch_slider.set(1.0)
        self.stretch_slider.pack(fill=tk.X, expand=True, padx=5)

        # 形态学处理
        self.morph_method_label = ttk.Label(self.param_frame, text="形态学处理:")
        self.morph_method_combo = ttk.Combobox(
            self.param_frame,
            values=["腐蚀", "膨胀", "开运算", "闭运算", "形态学梯度"],
            state="readonly", width=12)
        self.morph_method_combo.current(0)
        self.morph_method_combo.bind("<<ComboboxSelected>>", self.update_morph_params)

        # 形态学处理参数框架
        self.morph_param_frame = ttk.Frame(self.param_frame)

        # 核大小滑块
        self.kernel_size_frame = ttk.Frame(self.morph_param_frame)
        self.kernel_size_label = ttk.Label(self.kernel_size_frame, text="核大小: 3")
        self.kernel_size_label.pack(side=tk.LEFT)
        self.kernel_size_slider = ttk.Scale(
            self.kernel_size_frame, from_=1, to=15, style='Horizontal.TScale',
            command=lambda e: self.kernel_size_label.config(text=f"核大小: {int(float(e))}"))
        self.kernel_size_slider.set(3)
        self.kernel_size_slider.pack(fill=tk.X, expand=True, padx=5)

        # 迭代次数滑块
        self.iterations_frame = ttk.Frame(self.morph_param_frame)
        self.iterations_label = ttk.Label(self.iterations_frame, text="迭代: 1")
        self.iterations_label.pack(side=tk.LEFT)
        self.iterations_slider = ttk.Scale(
            self.iterations_frame, from_=1, to=10, style='Horizontal.TScale',
            command=lambda e: self.iterations_label.config(text=f"迭代: {int(float(e))}"))
        self.iterations_slider.set(1)
        self.iterations_slider.pack(fill=tk.X, expand=True, padx=5)

        # 处理按钮（使用绿色）
        self.process_button = ttk.Button(
            control_panel, text="应用处理", command=self.process_image, style='Process.TButton')
        self.process_button.pack(fill=tk.X, padx=5, pady=10)

        # 右侧图像显示区域
        image_display = ttk.Frame(main_frame)
        image_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        # 图像显示样式
        image_style = ttk.Style()
        image_style.configure('Image.TLabelframe', borderwidth=2, relief='solid')
        image_style.configure('Image.TLabel', background='white')

        # 原始图像
        self.original_group = ttk.LabelFrame(image_display, text="原始图像", style='Image.TLabelframe')
        self.original_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_label = ttk.Label(self.original_group, text="请加载图像", anchor="center",
                                        style='Image.TLabel')
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 处理后的图像
        self.processed_group = ttk.LabelFrame(image_display, text="处理后图像", style='Image.TLabelframe')
        self.processed_group.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.processed_label = ttk.Label(self.processed_group, text="处理后图像", anchor="center",
                                         style='Image.TLabel')
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 初始隐藏所有参数控件
        self.hide_all_params()

    def hide_all_params(self):
        """隐藏所有参数控件"""
        for widget in self.param_frame.winfo_children():
            widget.pack_forget()

    def update_visible_params(self, event=None):
        """根据选择的方法显示相应的参数控件"""
        self.hide_all_params()
        method = self.method_combo.get()

        if method == "对比度调整":
            self.contrast_frame.pack(fill=tk.X, padx=5, pady=5)
            self.process_button.pack(fill=tk.X, padx=5, pady=5)
        elif method == "去噪":
            self.denoise_method_label.pack(fill=tk.X, padx=5, pady=5)
            self.denoise_method_combo.pack(fill=tk.X, padx=5, pady=5)
            self.denoise_kernel_frame.pack(fill=tk.X, padx=5, pady=5)
            self.process_button.pack(fill=tk.X, padx=5, pady=5)
        elif method == "边缘检测":
            self.edge_method_label.pack(fill=tk.X, padx=5, pady=5)
            self.edge_method_combo.pack(fill=tk.X, padx=5, pady=5)
            if self.edge_method_combo.get() == "Canny":
                self.canny_thresh1_frame.pack(fill=tk.X, padx=5, pady=5)
                self.canny_thresh2_frame.pack(fill=tk.X, padx=5, pady=5)
            self.process_button.pack(fill=tk.X, padx=5, pady=5)
        elif method == "灰度拉伸":
            self.gray_stretch_method_label.pack(fill=tk.X, padx=5, pady=5)
            self.gray_stretch_method_combo.pack(fill=tk.X, padx=5, pady=5)
            if self.gray_stretch_method_combo.get() != "Otsu":
                self.stretch_frame.pack(fill=tk.X, padx=5, pady=5)
            self.process_button.pack(fill=tk.X, padx=5, pady=5)
        elif method == "形态学处理":
            self.morph_method_label.pack(fill=tk.X, padx=5, pady=5)
            self.morph_method_combo.pack(fill=tk.X, padx=5, pady=5)
            self.morph_param_frame.pack(fill=tk.X, padx=5, pady=5)
            self.process_button.pack(fill=tk.X, padx=5, pady=5)
        elif method != "原始图像":
            self.process_button.pack(fill=tk.X, padx=5, pady=5)

    def update_edge_params(self, event):
        """更新边缘检测方法的参数显示"""
        method = self.edge_method_combo.get()
        if method == "Canny":
            self.canny_thresh1_frame.pack(fill=tk.X, padx=5, pady=5)
            self.canny_thresh2_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.canny_thresh1_frame.pack_forget()
            self.canny_thresh2_frame.pack_forget()

    def update_stretch_params(self, event=None):
        """更新灰度拉伸方法的参数显示"""
        method = self.gray_stretch_method_combo.get()

        if method == "线性拉伸":
            # 线性拉伸不需要额外的参数，因此隐藏滑块
            self.stretch_frame.pack_forget()
        elif method == "对数拉伸" or method == "幂律拉伸":
            # 对数拉伸和幂律拉伸可能需要额外的参数，因此显示滑块
            self.stretch_frame.pack(fill=tk.X, padx=5, pady=5)

    def update_morph_params(self, event=None):
        """更新形态学处理方法的参数显示"""
        method = self.morph_method_combo.get()

        # 先清空参数框架
        for widget in self.morph_param_frame.winfo_children():
            widget.pack_forget()

        # 所有形态学操作都需要核大小
        self.kernel_size_frame.pack(fill=tk.X, padx=5, pady=5)

        # 只有腐蚀、膨胀、开运算、闭运算需要迭代次数
        if method in ["腐蚀", "膨胀", "开运算", "闭运算"]:
            self.iterations_frame.pack(fill=tk.X, padx=5, pady=5)

    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            title="打开图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif")])

        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # 显示原始图像
                self.display_image(self.original_image, self.original_label, "original")

                # 初始显示原始图像
                # self.processed_image = self.original_image.copy()
                self.display_image(self.processed_image, self.processed_label, "processed")
            else:
                messagebox.showerror("错误", "无法加载图像文件！")

    def save_image(self):
        """保存处理后的图像（原始尺寸）"""
        if self.processed_image is None:
            messagebox.showerror("警告", "没有可保存的图像！")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[
                ("PNG图像", "*.png"),
                ("JPEG图像", "*.jpg"),
                ("位图", "*.bmp"),
                ("所有文件", "*.*")
            ])

        if file_path:
            try:
                # 使用原始尺寸保存（非显示尺寸）
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("成功", f"图像已保存至：\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败：\n{str(e)}")

    def display_image(self, image, label, photo_type):
        """显示OpenCV图像到Tkinter标签"""
        if image is None:
            return

        # 调整图像大小以适应标签
        height, width = image.shape[:2]
        max_size = 400
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 转换颜色空间
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # 彩色图
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整大小
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 转换为PIL图像
        pil_image = Image.fromarray(image)

        # 转换为Tkinter PhotoImage
        photo = ImageTk.PhotoImage(pil_image)

        # 保存引用并更新标签
        if photo_type == "original":
            self.photo_original = photo
            label.config(image=self.photo_original)
        elif photo_type == "processed":
            self.photo_processed = photo
            label.config(image=self.photo_processed)
        elif photo_type == "hist_original":
            self.photo_hist_original = photo
            label.config(image=self.photo_hist_original)
        elif photo_type == "hist_processed":
            self.photo_hist_processed = photo
            label.config(image=self.photo_hist_processed)

        label.image = photo

    def start_locate_thread(self):
        """启动定位线程"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return

        threading.Thread(target=self.locate_mang, daemon=True).start()

    def locate_mang(self):
        """盲道定位核心逻辑"""
        try:
            processed = ImageProcessor.preprocess(self.original_image)

            # HSV颜色空间转换
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

            # 确保阈值数组类型正确
            lower = np.array(self.lower_hsv, dtype=np.uint8)
            upper = np.array(self.upper_hsv, dtype=np.uint8)
            # 颜色阈值处理
            mask = cv2.inRange(hsv, lower, upper)
            # 形态学优化
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
            dilated = cv2.dilate(opened, kernel, iterations=3)

            # 最大连通域提取
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError("未检测到有效盲道区域")

            max_contour = max(contours, key=cv2.contourArea)
            self.mask = np.zeros_like(dilated)
            cv2.drawContours(self.mask, [max_contour], -1, 255, -1)

            # 显示处理结果
            result_img = ImageProcessor.feature_extraction(self.original_image, self.mask)
            self.root.after(0, self.show_image, result_img, self.processed_label)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "处理错误", str(e))

    def show_segmentation(self):
        """显示分割结果"""
        if self.mask is None:
            messagebox.showwarning("警告", "请先进行定位处理")
            return

        segmented = cv2.bitwise_and(self.original_image, self.original_image, mask=self.mask)
        self.show_image(segmented, self.processed_label)

    def show_image(self, img, label_widget):
        """显示图像到指定标签（保持原始比例）"""
        # 保存原始处理结果用于后续保存
        self.processed_image = img.copy() if img is not None else None
        # 显示缩放处理
        h, w = img.shape[:2]
        max_size = (400, 300)  # 最大显示尺寸
        scale = min(max_size[0] / w, max_size[1] / h, 1)
        new_size = (int(w * scale), int(h * scale))

        # 生成显示用图像
        display_img = cv2.cvtColor(cv2.resize(img, new_size), cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(display_img))

        # 更新标签
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    def process_image(self):
        """处理图像"""
        if self.original_image is None:
            messagebox.showerror("警告", "请先加载图像！")
            return

        method = self.method_combo.get()

        try:
            if method == "原始图像":
                self.processed_image = self.original_image.copy()
            elif method == "灰度化":
                self.processed_image = ImageProcessor.to_gray(self.original_image)
            elif method == "对比度调整":
                value = int(self.contrast_slider.get())
                self.processed_image = ImageProcessor.contrast_adjust(self.original_image, value)
            elif method == "去噪":
                denoise_method = {
                    "高斯模糊": "gaussian",
                    "中值滤波": "median",
                    "双边滤波": "bilateral"
                }[self.denoise_method_combo.get()]
                kernel_size = int(self.denoise_kernel_slider.get())
                self.processed_image = ImageProcessor.nonoise(self.original_image, denoise_method, kernel_size)
            elif method == "边缘检测":
                edge_method = self.edge_method_combo.get().lower()
                if edge_method == "canny":
                    thresh1 = int(self.canny_thresh1_slider.get())
                    thresh2 = int(self.canny_thresh2_slider.get())
                    self.processed_image = ImageProcessor.edge_jiance(
                        self.original_image, edge_method, thresh1, thresh2)
                else:
                    self.processed_image = ImageProcessor.edge_jiance(
                        self.original_image, edge_method)
            elif method == "二值化":
                self.processed_image = ImageProcessor.to_binary(self.original_image, maxval=255)
            elif method == "形态学处理":
                morph_method = {
                    "腐蚀": "erode",
                    "膨胀": "dilate",
                    "开运算": "open",
                    "闭运算": "close",
                    "形态学梯度": "gradient"
                }[self.morph_method_combo.get()]

                kernel_size = int(self.kernel_size_slider.get())
                iterations = int(self.iterations_slider.get()) if morph_method != "gradient" else 1

                self.processed_image = ImageProcessor.xingtai_processing(
                    self.original_image,
                    method=morph_method,
                    kernel_size=kernel_size,
                    iterations=iterations
                )
            elif method == "灰度拉伸":
                stretch_method = {
                    "线性拉伸": "linear",
                    "对数拉伸": "logarithmic",
                    "幂律拉伸": "power-law"
                }[self.gray_stretch_method_combo.get()]

                if stretch_method == "linear":
                    self.processed_image = ImageProcessor.linear_stretch(self.original_image)
                elif stretch_method == "logarithmic":
                    stretch_param = float(self.stretch_slider.get())
                    self.processed_image = ImageProcessor.duishu_stretch(self.original_image, stretch_param)
                else:
                    stretch_param = float(self.stretch_slider.get())
                    self.processed_image = ImageProcessor.power_law_stretch(self.original_image, stretch_param)

            # 显示处理后的图像
            self.display_image(self.processed_image, self.processed_label, "processed")


        except Exception as e:
            messagebox.showerror("错误", f"图像处理失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()