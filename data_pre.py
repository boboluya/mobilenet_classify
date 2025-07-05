import tensorflow as tf
import numpy as np
import day3 as d3
import cv2
import matplotlib.pyplot as plt

img_pathes, labels = d3.get_img_pathes_and_labels()

train_size = int(0.8 * len(img_pathes))
val_size = len(img_pathes) - train_size


def preprocess_image(image_path, label):
    # 读取图片
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # 调整图片大小
    image = tf.image.resize(image, [224, 224])
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.02)

    # 归一化处理
    image = image / 255.0
    label = tf.cast(label, tf.float32)

    return image, label


print(f"图片数量: {len(img_pathes)}")
print(f"标签数量: {len(labels)}")

datasets1 = tf.data.Dataset.from_tensor_slices((img_pathes, labels))

datasets1 = datasets1.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
# datasets1 = datasets1.batch(1)

# # 可视化
# for image_batch, label_batch in datasets1.take(5):
#     image_np = image_batch[0].numpy()  # 转换为 numpy 数组
#     label = label_batch[0].numpy()

#     h, w = 224, 224  # 因为已经 resize 成 224x224
#     x, y, bw, bh = label
#     x1 = int(x * w)
#     y1 = int(y * h)
#     x2 = int((x + bw) * w)
#     y2 = int((y + bh) * h)

#     # 画框
#     image_np = np.array(image_np * 255, dtype=np.uint8)
#     image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)

#     plt.imshow(image_np)
#     plt.title(f"Label: {label}")
#     plt.axis('off')
#     plt.show()
#     input("按回车显示下一张图片...")

# 增加批次大小以更好地估计梯度
batch_size = 64
train_dataset = datasets1.take(train_size).shuffle(1000).batch(batch_size)
val_dataset = datasets1.skip(train_size).batch(batch_size)

# 使用更轻量级的模型或适当减少参数
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

# 冻结更多层，只训练顶部的一小部分
base_model.trainable = True
for layer in base_model.layers[:-1]:  # 只训练最后1层
    layer.trainable = False

# 简化模型架构，减少参数
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        # 减少中间层的神经元数量
        tf.keras.layers.Dense(64, activation="relu"),
        # 增加更强的Dropout防止过拟合
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation="sigmoid"),
    ]
)

# L2正则化，增加训练中的泛化能力
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=train_size // batch_size * 2,  # 每两个epochs衰减一次
        decay_rate=0.9,
    )
)

# 增加验证集上的耐心值，让模型有更多机会找到更好的泛化点
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

# 增加学习率衰减和训练策略
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
)

model.compile(optimizer="adamw", loss="mse", metrics=["mae"])

# 增加训练轮次
model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=[reduce_lr, early_stop, checkpoint],
)

# 保存模型
model.save("face_detection_model.keras")
model.summary()
