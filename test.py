import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
import os

class MobileNetV3FaceDetector:
    """基于MobileNetV3的人脸检测模型，输出边界框坐标"""
    
    def __init__(self, input_shape=(224, 224, 3), max_faces=10):
        self.input_shape = input_shape
        self.max_faces = max_faces  # 最大检测人脸数量
        self.model = None
        
    def build_model(self):
        """构建基于MobileNetV3的人脸检测模型"""
        # 使用MobileNetV3作为骨干网络
        base_model = MobileNetV3Large(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # 冻结部分基础层
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # 输入层
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # 骨干网络特征提取
        x = base_model(inputs, training=False)
        
        # 添加检测头
        # 全局平均池化
        x = layers.GlobalAveragePooling2D()(x)
        
        # 特征映射
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 人脸数量预测分支
        num_faces = layers.Dense(1, activation='sigmoid', name='num_faces')(x)
        
        # 边界框坐标预测分支 (每个人脸4个坐标: x, y, w, h)
        # 输出 max_faces * 4 个值
        bbox_coords = layers.Dense(self.max_faces * 4, activation='sigmoid', name='bbox_coords')(x)
        
        # 置信度预测分支 (每个人脸一个置信度)
        confidences = layers.Dense(self.max_faces, activation='sigmoid', name='confidences')(x)
        
        # 构建模型
        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                'num_faces': num_faces,
                'bbox_coords': bbox_coords,
                'confidences': confidences
            }
        )
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'num_faces': 'mse',
                'bbox_coords': 'mse', 
                'confidences': 'binary_crossentropy'
            },
            loss_weights={
                'num_faces': 1.0,
                'bbox_coords': 2.0,
                'confidences': 1.0
            },
            metrics={
                'num_faces': 'mae',
                'bbox_coords': 'mae',
                'confidences': 'accuracy'
            }
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """预处理输入图像"""
        if isinstance(image, str):
            # 如果输入是文件路径
            image = tf.io.read_file(image)
            image = tf.image.decode_image(image, channels=3)
        
        # 调整大小
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]])
        
        # 归一化到[0,1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # 增加批次维度
        image = tf.expand_dims(image, 0)
        
        return image
    
    def postprocess_predictions(self, predictions, confidence_threshold=0.5):
        """后处理预测结果"""
        num_faces = predictions['num_faces'][0][0]
        bbox_coords = predictions['bbox_coords'][0]
        confidences = predictions['confidences'][0]
        
        # 估计人脸数量
        estimated_faces = int(np.round(num_faces * self.max_faces))
        estimated_faces = max(0, min(estimated_faces, self.max_faces))
        
        # 解析边界框坐标
        bboxes = bbox_coords.reshape(-1, 4)  # (max_faces, 4)
        
        # 过滤有效的检测结果
        valid_detections = []
        for i in range(self.max_faces):
            confidence = confidences[i]
            if confidence > confidence_threshold:
                bbox = bboxes[i]
                # 转换坐标格式 (x, y, w, h) -> (x1, y1, x2, y2)
                x, y, w, h = bbox
                x1 = int(x * self.input_shape[1])
                y1 = int(y * self.input_shape[0])
                x2 = int((x + w) * self.input_shape[1])
                y2 = int((y + h) * self.input_shape[0])
                
                valid_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(confidence),
                    'face_id': i
                })
        
        return {
            'estimated_faces': estimated_faces,
            'detections': valid_detections,
            'raw_predictions': predictions
        }
    
    def detect_faces(self, image, confidence_threshold=0.5):
        """检测图像中的人脸边界框"""
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_model()")
        
        # 预处理
        processed_image = self.preprocess_image(image)
        
        # 预测
        predictions = self.model.predict(processed_image, verbose=0)
        
        # 后处理
        results = self.postprocess_predictions(predictions, confidence_threshold)
        
        return results
    
    def visualize_detections(self, image, results, figsize=(12, 8)):
        """可视化检测结果"""
        if isinstance(image, str):
            # 如果是文件路径，读取图像
            image_array = plt.imread(image)
        else:
            image_array = image
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image_array)
        
        # 绘制边界框
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            face_id = detection['face_id']
            
            # 绘制矩形框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x1, y1-5, f'Face {face_id}: {confidence:.2f}', 
                   color='red', fontsize=12, weight='bold')
        
        ax.set_title(f'检测到 {len(results["detections"])} 个人脸 (估计: {results["estimated_faces"]})')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_training_data(self, batch_size=32):
        """生成用于演示的合成训练数据"""
        # 这里生成合成数据用于演示
        # 实际应用中需要真实的标注数据
        
        images = np.random.random((batch_size, *self.input_shape))
        
        # 模拟标签
        num_faces = np.random.randint(0, self.max_faces, size=(batch_size, 1)) / self.max_faces
        
        # 模拟边界框坐标 (归一化坐标)
        bbox_coords = np.random.random((batch_size, self.max_faces * 4))
        
        # 模拟置信度
        confidences = np.random.random((batch_size, self.max_faces))
        
        return images, {
            'num_faces': num_faces,
            'bbox_coords': bbox_coords,
            'confidences': confidences
        }

def demo_mobilenetv3_face_detection():
    """演示MobileNetV3人脸检测模型"""
    print("=== MobileNetV3 人脸检测模型演示 ===")
    
    # 初始化检测器
    detector = MobileNetV3FaceDetector(max_faces=5)
    
    # 构建模型
    print("构建MobileNetV3人脸检测模型...")
    model = detector.build_model()
    
    print("模型结构:")
    model.summary()
    
    # 生成测试数据
    print("\n生成合成测试图像...")
    test_image = np.random.randint(0, 255, (*detector.input_shape,), dtype=np.uint8)
    
    # 检测人脸
    print("检测人脸边界框...")
    results = detector.detect_faces(test_image, confidence_threshold=0.3)
    
    print(f"估计人脸数量: {results['estimated_faces']}")
    print(f"检测到的人脸: {len(results['detections'])}")
    
    for i, detection in enumerate(results['detections']):
        bbox = detection['bbox']
        confidence = detection['confidence']
        print(f"  人脸 {i+1}: 边界框 {bbox}, 置信度: {confidence:.3f}")
    
    # 可视化结果
    print("\n可视化检测结果...")
    try:
        detector.visualize_detections(test_image, results)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    return detector

def training_demo():
    """训练演示"""
    print("\n=== 训练演示 ===")
    
    detector = MobileNetV3FaceDetector(max_faces=3)
    model = detector.build_model()
    
    # 生成合成训练数据
    print("生成合成训练数据...")
    X_train, y_train = detector.generate_training_data(batch_size=100)
    X_val, y_val = detector.generate_training_data(batch_size=20)
    
    # 训练模型
    print("开始训练...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    print("训练完成!")
    return model, history

if __name__ == "__main__":
    # 运行演示
    detector = demo_mobilenetv3_face_detection()
    
    print("\n" + "="*50)
    
    # 运行训练演示
    model, history = training_demo()
    
    print("\n模型特性:")
    print("1. 基于MobileNetV3轻量级架构")
    print("2. 直接输出人脸边界框坐标")
    print("3. 支持多人脸检测(可配置最大数量)")
    print("4. 输出置信度分数")
    print("5. 端到端训练")
    
    print("\n使用方法:")
    print("detector = MobileNetV3FaceDetector(max_faces=10)")
    print("detector.build_model()")
    print("results = detector.detect_faces(image)")
    print("detector.visualize_detections(image, results)")