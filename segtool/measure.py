import math
import cv2
import numpy as np
import random
from queue import Queue
from pathlib import Path
import os
from tqdm import tqdm

midline_color = 5
seed = 5277
# 设置随机种子
# np.random.seed(seed)
# random.seed(seed)
patch_number = 0


def crack_measure_single(img, xmin, ymin, xmax, ymax):
    avg_width = 0
    min_width = 99999
    max_width = 0
    # print(f'submat shape: {ymax - ymin} x {xmax - xmin}')
    sub_image = img[ymin:ymax, xmin:xmax]
    """
    find 轮廓
    第二个参数表示轮廓的检索模式，有四种：
        cv2.RETR_EXTERNAL表示只检测外轮廓
        cv2.RETR_LIST检测的轮廓不建立等级关系
        cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        cv2.RETR_TREE建立一个等级树结构的轮廓。
    第三个参数method为轮廓的近似办法
        cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    """
    contours, hierarchy = cv2.findContours(sub_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) > 5:
    #     return 0, 0, 0
    # print(contours)

    contour_img = cv2.cvtColor(sub_image, cv2.COLOR_GRAY2RGB)

    mask = np.zeros(sub_image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # cv2.imwrite(f"./work_dir/{xmin}_{ymin}mask.jpg", mask)

    mp = dict()
    for i in range(0, 256):
        mp[i] = []
    # dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3, cv2.DIST_LABEL_PIXEL)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 0, cv2.DIST_LABEL_PIXEL)
    trans_img = cv2.convertScaleAbs(dt)
    cv2.normalize(trans_img, trans_img, 0, 255, cv2.NORM_MINMAX)
    for i in range(trans_img.shape[1]):
        for j in range(trans_img.shape[0]):
            mp[trans_img[j][i]].append((j, i))

    lines = []
    sub_image = cv2.ximgproc.thinning(sub_image, cv2.ximgproc.THINNING_ZHANGSUEN)

    for i in range(sub_image.shape[1]):
        for j in range(sub_image.shape[0]):
            if sub_image[j][i] >= 200:
                # cv2.circle(contour_img, (i, j), 1, (0, 0, 255), 1)
                lines.append((j, i))

    endpoints = []
    for (j, i) in lines:
        sub_mat = sub_image[j - 1:j + 1 + 1, i - 1:i + 1 + 1]
        cnt = 0
        for ii in range(sub_mat.shape[1]):
            for jj in range(sub_mat.shape[0]):
                if sub_mat[jj][ii] >= 200:
                    cnt += 1
        if cnt <= 2:
            # cv2.circle(contour_img, (i, j), 2, (0, 255, 0), 2)
            endpoints.append((j, i))

    directions = [
        (1, -1), (1, 0), (1, 1),
        (0, -1), (0, 1),
        (-1, -1), (-1, 0), (-1, 1)
    ]
    for j, i in endpoints:
        queue = [(j, i)]
        f, r = 0, 0
        while f <= r and len(queue) < 9:
            x, y = queue[f]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= sub_image.shape[0]:
                    continue
                if ny < 0 or ny >= sub_image.shape[1]:
                    continue
                if sub_image[nx][ny] >= 200 and queue.count((nx, ny)) == 0:
                    queue.append((nx, ny))
                    r += 1
            f += 1
        dirs = np.array(queue)
        line = cv2.fitLine(dirs, cv2.DIST_L2, 0, 0.01, 0.01)

        j_ = int(j + line[0][0] * 5 + 0.5)
        i_ = int(i + line[1][0] * 5 + 0.5)

        if (j_ - queue[-1][0]) ** 2 + (i_ - queue[-1][1]) ** 2 > (j - queue[-1][0]) ** 2 + (i - queue[-1][1]) ** 2:
            for l in range(1, 100):
                jj = int(j + line[0][0] * l + 0.5)
                ii = int(i + line[1][0] * l + 0.5)
                if jj < 0 or jj >= sub_image.shape[0]:
                    break
                if ii < 0 or ii >= sub_image.shape[1]:
                    break
                if mask[jj][ii] == 0:
                    break
                sub_image[jj][ii] = 255
                cv2.circle(contour_img, (ii, jj), 1, (0, 0, 255), 1)
        else:
            for l in range(1, 100):
                jj = int(j - line[0][0] * l + 0.5)
                ii = int(i - line[1][0] * l + 0.5)
                if jj < 0 or jj >= sub_image.shape[0]:
                    break
                if ii < 0 or ii >= sub_image.shape[1]:
                    break
                if mask[jj][ii] == 0:
                    break
                sub_image[jj][ii] = 255
                cv2.circle(contour_img, (ii, jj), 1, (0, 0, 255), 1)

    trajectory = []
    for i in range(sub_image.shape[1]):
        for j in range(sub_image.shape[0]):
            if sub_image[j][i] >= 200:
                cv2.circle(contour_img, (i, j), 1, (0, 0, 255), 1)
                trajectory.append((j, i))

    widths = []
    for p in trajectory:
        min_distance = 99999
        for cnt in contours:
            distance = cv2.pointPolygonTest(cnt, (p[1], p[0]), True)
            min_distance = min(min_distance, np.abs(distance))
        widths.append(min_distance)
    avg_width = 0 if len(widths) == 0 else sum(widths) / len(widths) * 2
    max_width = 0 if len(widths) == 0 else max(widths) * 2
    endpoints_for_calc = []
    for (j, i) in trajectory:
        sub_mat = sub_image[max(0, j - 1):min(j + 1 + 1, sub_image.shape[0] - 1),
                  max(0, i - 1):min(i + 1 + 1, sub_image.shape[1] - 1)]
        cnt = 0
        pts = []
        for ii in range(sub_mat.shape[1]):
            for jj in range(sub_mat.shape[0]):
                if sub_mat[jj][ii] >= 200:
                    cnt += 1
                    pts.append((jj, ii))
        if cnt <= 2:
            cv2.circle(contour_img, (i, j), 2, (0, 255, 0), 1)
            endpoints_for_calc.append((j, i))
        elif cnt == 3:
            sz = sub_mat.shape[0] * sub_mat.shape[1]
            dis = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dis.append((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            dis.sort()
            # if dis[0] == dis[1] and dis[1] == 1 and dis[2] == 2:
            #     endpoints_for_calc.append((j, i))

    def bfs(sx, sy, tag):
        q = Queue()
        q.put((sx, sy, tag))
        while q.empty() is False:
            x, y, tag = q.get()
            for dx, dy in [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (1, 1), (-1, -1), (-1, 1)
            ]:
                nx, ny = x + dx, y + dy
                if nx >= sub_image.shape[0] or ny >= sub_image.shape[1]:
                    continue
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if trajectory.count((nx, ny)) == 0:
                    continue
                if (nx, ny) in visited:
                    if visited[(nx, ny)] > 0:
                        continue
                    visited['len'] += distance
                    # if caount==0 and patch_number==9:
                    #     print(f'{x}, {y} => {nx}, {ny}')
                    continue

                visited['len'] += distance
                visited[(nx, ny)] = 0
                # print(f'{x}, {y} -> {nx}, {ny}')
                q.put((nx, ny, tag))

    idx = 1
    visited = dict()
    visited['len'] = 0
    for x, y in endpoints_for_calc:
        if (x, y) in visited:
            continue
        visited[(x, y)] = idx
        bfs(x, y, idx)
        idx += 1

    return avg_width, max_width, visited['len']


def denoise(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # 找到最大连通组件
    largest_label = 1  # 默认背景标签
    max_area = 0

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[label, cv2.CC_STAT_AREA]
            largest_label = label

    # 创建一个新的二值图像，只包含最大连通组件
    output_image = np.zeros(image.shape, dtype=np.uint8)
    output_image[labels == largest_label] = 255  # 设置最大连通组件为白色

    return output_image


def error_calc(image, label):
    image = np.pad(image, pad_width=5, mode='constant', constant_values=0)
    label = np.pad(label, pad_width=5, mode='constant', constant_values=0)
    image = denoise(image)
    label = denoise(label)
    res = []
    a, b, c = crack_measure_single(image, 0, 0, image.shape[1], image.shape[0], caount=0)
    res.append(a)
    res.append(b)
    res.append(c)
    return res


def preprocess(img):
    # 闭操作，连接一些细小的断裂，可以不要
    kernel = np.ones((3, 3), np.uint8)
    cvclose = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭操作
    return cvclose


def findBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv_contours = []
    box = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓面积
        length = cv2.arcLength(contours[i], True)  # 计算轮廓周长
        if length <= 0:
            length = 1e-4
        roundness = (4 * math.pi * area) / (length * length)  # 圆形度
        area = cv2.contourArea(contours[i])
        if roundness > 0.3 or area < 20:  # 这里设置去除的小面积区域的参数
            cv_contours.append(contours[i])
        else:
            x, y, w, h = cv2.boundingRect(contours[i])
            box.append([x, y, x + w, y + h])
    cv2.fillPoly(img, cv_contours, (0, 0, 0))
    return img, box


def measure(mask: np.ndarray):
    """
    input：二值图/灰度图
    output：一个二维list，list[i]表示第i个裂缝的信息，依次是：
                        裂缝边界框的位置信息，以(xmin, ymin, xmax, ymax)的形式给出
                        裂缝的长度
                        裂缝的平均宽度
                        裂缝的最大宽度
    """
    result = []
    # print('shape of mask:', mask.shape)
    # print('type of mask:', mask.dtype)
    # 二值化
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    # 根据裂缝划分bounding box
    mask, boxes = findBox(mask)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        avg_width, max_width, length = crack_measure_single(mask, xmin=x1, ymin=y1, xmax=x2, ymax=y2)
        tmp_dict = dict()
        tmp_dict['box']=box
        tmp_dict['length']=length
        tmp_dict['avg_width']=avg_width
        tmp_dict['max_width']=max_width
        result.append(tmp_dict)
    return result

if __name__ == '__main__':
    mask_dir = '/nfs/wj/result/0825/unet2/dam'
    paths = [path for path in Path(mask_dir).glob('*.*')]
    paths.sort()
    for i, path in enumerate(paths):
        if i!=109:
            continue
        print(path.stem)
        mask_path = os.path.join(mask_dir, path.name)
        mask = cv2.imread(mask_path, 0)
        result = measure(mask=mask)
        print(result)
        break
    pass
