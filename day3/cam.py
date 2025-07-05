import tensorflow as tf
import cv2
import numpy as np

# 加载模型
model = tf.keras.models.load_model("face_detection_model.keras")

# 定义预测函数
def predict_face_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # [1, 224, 224, 3]
    prediction = model.predict(img, verbose=0)[0]
    return prediction  # [x, y, w, h]，归一化坐标

# 启动摄像头
cap = cv2.VideoCapture(0)  # 默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 q 退出程序")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 预测人脸框
    pred = predict_face_frame(frame)
    x, y, bw, bh = pred
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)

    # 画框
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
