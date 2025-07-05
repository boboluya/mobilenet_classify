import os
import cv2

BASIC_PATH = "D:\\faces\\train\\"
TXT_PATH = BASIC_PATH + "train.txt"


def get_img_pathes_and_labels():
    img_pathes = []
    labels = []

    with open(TXT_PATH, "r") as f:
        lines = f.readlines()

    line = 0
    while line < len(lines):
        # 当前图片人脸数量
        num_faces = int(lines[line + 1].strip())

        # 跳过人脸数量 > 1 的图像
        if num_faces > 1:
            line += 2 + num_faces if num_faces > 0 else 3
            continue
        
        if num_faces == 0:
            # 如果没有人脸，直接跳过
            line += 3
            continue

        # 添加图片路径
        img_path = os.path.join(BASIC_PATH, lines[line].strip().split(" ")[0])
        img_pathes.append(img_path)

        # 添加对应标签
        label = []
        face_box = lines[line + 2].strip().split(" ")
        box = [float(coord) for coord in face_box[:4]]
        # 计算比例
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            box = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        label = box
        labels.append(label)

        # 更新读取位置
        line += 2 + num_faces if num_faces > 0 else 3

        # if len(img_pathes) >= 2000:
        #     break

    return img_pathes, labels



if __name__ == "__main__":
    img_pathes, labels = get_img_pathes_and_labels()
    print(img_pathes)
    print(labels)
    print(f"总共图片数量: {len(img_pathes)}")
    print(f"总共标签数量: {len(labels)}")
    # 随机挑选5张图片画框查看是否正确
    import random
    import matplotlib.pyplot as plt
    for i in random.sample(range(len(img_pathes)), 5):
        img = cv2.imread(img_pathes[i])
        h, w, _ = img.shape
        box = labels[i]
        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int((box[0] + box[2]) * w)
        y2 = int((box[1] + box[3]) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
