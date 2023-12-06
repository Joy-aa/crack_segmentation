import math
import cv2 as cv
import numpy as np
import random
from queue import Queue
from pathlib import Path
import os
from datetime import datetime
import csv

midline_color = 5
count=0
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
    contours, hierarchy = cv.findContours(sub_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print(contours)

    contour_img = cv.cvtColor(sub_image, cv.COLOR_GRAY2RGB)

    mask = np.zeros(sub_image.shape, np.uint8)
    cv.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}mask.jpg", mask)

    mp = dict()
    for i in range(0, 256):
        mp[i] = []
    # dt = cv.distanceTransform(mask, cv.DIST_L2, 3, cv.DIST_LABEL_PIXEL)
    dt = cv.distanceTransform(mask, cv.DIST_L2, 0, cv.DIST_LABEL_PIXEL)
    trans_img = cv.convertScaleAbs(dt)
    cv.normalize(trans_img, trans_img, 0, 255, cv.NORM_MINMAX)
    for i in range(trans_img.shape[1]):
        for j in range(trans_img.shape[0]):
            mp[trans_img[j][i]].append((j, i))

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}distance_transform.jpg", trans_img)

    lines = []
    for idx in range(1, 256):
        random.shuffle(mp[idx])
        for j, i in mp[idx] + mp[idx]:
            sub_image[j][i] = 0
            cnts, hrc = cv.findContours(sub_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(cnts) - len(contours) > 0 and idx > 100:
                sub_image[j][i] = 255
            else:
                sub_image[j][i] = 0

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}line.jpg", sub_image)

    for i in range(sub_image.shape[1]):
        for j in range(sub_image.shape[0]):
            if sub_image[j][i] == 255:
                lines.append((j, i))

    endpoints = []
    for (j, i) in lines:
        sub_mat = sub_image[j - 1:j + 1 + 1, i - 1:i + 1 + 1]
        cnt = 0
        for ii in range(sub_mat.shape[1]):
            for jj in range(sub_mat.shape[0]):
                if sub_mat[jj][ii] == 255:
                    cnt += 1
        if cnt <= 2:
            cv.circle(contour_img, (i, j), 2, (0, 255, 0), 2)
            endpoints.append((j, i))

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}line-color-before.jpg", contour_img)
    # cv.imwrite('test.jpg', sub_image)

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
                if sub_image[nx][ny] == 255 and queue.count((nx, ny)) == 0:
                    queue.append((nx, ny))
                    r += 1
            f += 1
        dirs = np.array(queue)
        line = cv.fitLine(dirs, cv.DIST_L2, 0, 0.01, 0.01)

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
                cv.circle(contour_img, (ii, jj), 1, (0, 0, 255), 1)
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
                cv.circle(contour_img, (ii, jj), 1, (0, 0, 255), 1)

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}line.jpg", sub_image)

    trajectory = []
    for i in range(sub_image.shape[1]):
        for j in range(sub_image.shape[0]):
            if sub_image[j][i] == 255:
                cv.circle(contour_img, (i, j), 1, (0, 0, 255), 1)
                trajectory.append((j, i))

    # cv.imwrite(f"./work_dir/{xmin}_{ymin}line-color.jpg", contour_img)

    widths = []
    for p in trajectory:
        min_distance = 99999
        for cnt in contours:
            distance = cv.pointPolygonTest(cnt, (p[1], p[0]), True)
            min_distance = min(min_distance, np.abs(distance))
        widths.append(min_distance)
    avg_width = 0 if len(widths) == 0 else sum(widths) / len(widths) * 2
    max_width = 0 if len(widths) == 0 else max(widths) * 2
    # print(avg_width, max_width)
    # print(f'trajectory: {trajectory}')
    endpoints_for_calc = []
    for (j, i) in trajectory:
        sub_mat = sub_image[max(0, j - 1):min(j + 1 + 1, sub_image.shape[0] - 1),
                  max(0, i - 1):min(i + 1 + 1, sub_image.shape[1] - 1)]
        cnt = 0
        pts = []
        for ii in range(sub_mat.shape[1]):
            for jj in range(sub_mat.shape[0]):
                if sub_mat[jj][ii] == 255:
                    cnt += 1
                    pts.append((jj, ii))
        if cnt <= 2:
            cv.circle(contour_img, (i, j), 2, (0, 255, 0), 2)
            endpoints_for_calc.append((j, i))
        elif cnt == 3:
            sz = sub_mat.shape[0] * sub_mat.shape[1]
            dis = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dis.append((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            dis.sort()
            if dis[0] == dis[1] and dis[1] == 1 and dis[2] == 2:
                endpoints_for_calc.append((j, i))

    def dfs(x, y, tag, visited):
        # print(f'travel {x} {y} in {tag}')

        for dx, dy in [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (1, 1), (-1, -1), (-1, 1)
        ]:
            nx, ny = x + dx, y + dy
            distance = np.sqrt(dx ** 2 + dy ** 2)
            if trajectory.count((nx, ny)) == 0:
                continue
            if (nx, ny) in visited:
                if visited[(nx, ny)] == tag:
                    continue
                visited['len'] += distance
                # print(f'{x}, {y} => {nx}, {ny}')
                break

            visited['len'] += distance
            visited[(nx, ny)] = 0
            # print(f'{x}, {y} -> {nx}, {ny}')
            dfs(nx, ny, tag, visited)
            break

    # print(endpoints_for_calc)

    idx = 0
    visited = dict()
    visited['len'] = 0
    for x, y in endpoints_for_calc:
        if (x, y) in visited:
            continue
        visited[(x, y)] = idx
        dfs(x, y, idx, visited)
        idx += 1

    # print(visited['len'])

    return avg_width, max_width, visited['len']


def denoise(image):
    mx = []
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j][i] == 255:
                image[j][i] = 0
                q = Queue()
                v = []
                q.put((j, i))
                v.append((j, i))
                while q.empty() is False:
                    p = q.get()
                    for dx, dy in [
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, -1), (1, 1), (-1, -1), (-1, 1)
                    ]:
                        pt = (p[0] + dx, p[1] + dy)
                        # for k in range(2):
                        if pt[0] < 0 or pt[0] >= image.shape[0] or pt[1] < 0 or pt[1] >= image.shape[1]:
                            continue
                        if image[pt[0]][pt[1]] == 255:
                            image[pt[0]][pt[1]] = 0
                            q.put((pt[0], pt[1]))
                            v.append((pt[0], pt[1]))
                if len(v) > len(mx):
                    mx.clear()
                    for t in v:
                        mx.append(t)
    for pt in mx:
        image[pt[0]][pt[1]]=255

    return image


def fill_in(image, maxValue=1):
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (1, 1), (-1, -1), (-1, 1)
    ]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cnt = 0
            for dire in directions:
                if i + dire[0] < 0 or i + dire[0] >= image.shape[0] or j + dire[1] < 0 or j + dire[1] >= image.shape[1]:
                    cnt += 1
                    continue
                if image[i + dire[0]][j + dire[1]] > 50:
                    cnt += 1
            if cnt >= 7:
                image[i][j] = maxValue
    return image
    pass


def error_calc(image, label):
    global count
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    label = np.pad(label, pad_width=1, mode='constant', constant_values=0)
    _, image = cv.threshold(image, 50, 255, cv.THRESH_BINARY)
    _, label = cv.threshold(label, 0, 255, cv.THRESH_BINARY)
    image = fill_in(image, 255)
    label = fill_in(label, 255)
    image = denoise(image)
    label = denoise(label)
    res = []
    a, b, c = crack_measure_single(image, 0, 0, image.shape[1], image.shape[0])
    res.append(a)
    res.append(b)
    res.append(c)
    a, b, c = crack_measure_single(label, 0, 0, label.shape[1], label.shape[0])
    res.append(a)
    res.append(b)
    res.append(c)
    for i in range(3):
        if res[i+3] == 0:
            res.append(1)
        elif res[i] == 0:
            # cv.imwrite('./work_dir/label'+str(count)+'.png', label)
            # cv.imwrite('./work_dir/image'+str(count)+'.jpg', image)
            # count += 1
            res.append(0)
        else:
            res.append(res[i + 3] / res[i])
    return res


if __name__ == '__main__':
    label_dir = '/mnt/nfs/wj/data/new_label'
    # label_dir = '/nfs/wj/data/new_label'
    mask_dir = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/test_result'
    # mask_dir = '/mnt/nfs/wj/test_result'

    paths = [path for path in Path(mask_dir).glob('*.*')]
    # path = paths[0]
    d = datetime.today()
    datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./work_dir', exist_ok=True)
    with open(os.path.join('./work_dir', str(d)+'.csv'), 'a', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_name", "avg_width_error", "max_width_error", "length_error"])
        total_length_error = 0
        for path in paths:
            print(path)
            label_path = os.path.join(label_dir, path.stem + '.png')
            mask_path = os.path.join(mask_dir, path.name)
            filepath = os.path.join('/mnt/nfs/wj/result-stride_0.7/Jun02_06_33_42/box', path.stem+'.txt')
            # filepath = os.path.join('/nfs/wj/result-stride_0.7/Jun02_06_33_42/box', path.stem + '.txt')
            boxes = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for data in f.readlines():
                    box = data.split(' ')[:-1]
                    boxes.append(box)
            label = cv.imread(label_path, 0)
            mask = cv.imread(mask_path, 0)
            avg_width_error=0
            max_width_error=0
            length_error = 0
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                label_pat = label[y1:y2, x1:x2]
                mask_pat = mask[y1:y2, x1:x2]
                res = error_calc(mask_pat, label_pat)
                avg_width_error += abs(1 - res[-3]) / len(boxes)
                max_width_error += abs(1 - res[-2]) / len(boxes)
                length_error += abs(1 - res[-1]) / len(boxes)
            writer.writerow([path.name, avg_width_error, max_width_error, length_error])
            total_length_error += length_error / len(paths)
        print(total_length_error)
        writer.writerow([total_length_error])