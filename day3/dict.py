import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("face_detection_model.keras")

def predict_face(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # 调整大小为224x224
    image = tf.cast(image, tf.float32)  # 转换为float32类型
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)  # [1, 224, 224, 3]

    predictions = model.predict(image)  # 直接预测即可
    return predictions[0]  # 取出纯数值


def visualize_prediction(image_path, box):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    x, y, bw, bh = box
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = "D:\\faces\\train\\32--Worker_Laborer\\32_Worker_Laborer_Worker_Laborer_32_38.jpg"
    prediction = predict_face(image_path)
    print(f"预测结果: {prediction}")
    visualize_prediction(image_path, prediction)
