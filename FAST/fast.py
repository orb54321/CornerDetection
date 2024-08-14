import numpy as np
import cv2
import matplotlib.pyplot as plt

N = 9
radius = 3
row_coord_on_circle = np.array([0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0])
col_coord_on_circle = np.array([3, 4, 5, 6, 6, 6, 5, 4, 3, 2, 1, 0, 0, 0, 1, 2])

def FastCornerDetect(input, threshold):
    score_array = np.zeros(input.shape).astype(np.float32)
    img = input.copy()
    img = img.astype(np.float32)
    row_start = radius
    col_start = radius
    row_end, col_end = img.shape[0] - radius, img.shape[1] - radius
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            detect_zone = img[row - radius:row + radius + 1, col - radius:col + radius + 1]
            data_on_circle = (detect_zone[row_coord_on_circle, col_coord_on_circle]).astype(np.float32)
            data_center = img[row, col]
            condition1 = ((data_center - data_on_circle) > threshold).astype(int)  # 中间点大于周围点，为1
            condition2 = ((data_on_circle - data_center) < threshold).astype(int)  # 中间点小于周围点，为1
            # if condition1[0] + condition1[8] >= 1 and condition1[4] + condition1[12] >= 1:
            if (condition1[0] >= 1 or condition1[8] >= 1) and (condition1[4] >= 1 or condition1[12] >= 1):
                cond1 = condition1.copy()
                cond1 = np.concatenate((cond1, cond1[0: N - 1]), axis=0)  # 拼接，cond1中最后一个像素也需要进行判断，所以给它后面加了八个像素，使得可以判断圆周上是否有连续N个点，其像素值与中心位置像素值之差大于t（或小于-t）
                cond1_str = ' '.join(str(i) for i in cond1)
                cond1_str = cond1_str.replace('0', ' ')
                cond1_str = cond1_str.split()
                cond1_len = np.array([len(j) for j in cond1_str])
                if np.max(cond1_len) >= N:
                    score = np.sum(np.abs(data_center-data_on_circle))
                    score_array[row, col] = score
                    continue
            # if condition2[0] + condition2[8] >= 1 and condition2[4] + condition2[12] >= 1:
            if (condition1[0] >= 1 or condition1[8] >= 1) and (condition1[4] >= 1 or condition1[12] >= 1):
                cond2 = condition2.copy()
                cond2 = np.concatenate((cond2, cond2[0: N - 1]), axis=0)
                cond2_str = ''.join(str(i) for i in cond2)
                cond2_str = cond2_str.replace('0', ' ')
                cond2_str = cond2_str.split()
                cond2_len = np.array([len(j) for j in cond2_str])
                if np.max(cond2_len) >= N:
                    score = np.sum(np.abs(data_center-data_on_circle))
                    score_array[row, col] = score
    return score_array


def nms(score_array, kernel = 3):
    out = score_array.copy()
    row_start = int(kernel / 2)
    col_start = int(kernel / 2)
    row_end, col_end = out.shape[0] - kernel, out.shape[1] - kernel
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if out[row, col] == 0:
                continue
            max_score = np.max(out[row - int(kernel / 2) : row + int(kernel / 2) + 1, col - int(kernel / 2):col + int(kernel / 2) + 1])
            if out[row, col] < max_score:
                out[row, col] = 0
            else:
                out[row, col] = 255

    return out

def main():
    img = cv2.imread('./data/input/input3.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (256, 256))
    score_array = FastCornerDetect(img, 50)
    img_without_nms = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_without_nms[score_array > 0] = [0, 0, 255]
    plt.figure()
    plt.title('FAST corner detection')
    cv2.imwrite('./data/output/fast_corner_without_nms.jpg', img_without_nms)
    plt.imshow(img_without_nms)

    out = nms(score_array)
    print(out)
    img_with_nms = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_with_nms[out > 0] = [0, 0, 255]
    plt.figure()
    plt.title('FAST corner detection after NMS')
    plt.imshow(img_with_nms, cmap='gray')
    cv2.imwrite('./data/output/fast_corner_with_nms.jpg', img_with_nms)
    plt.show()

if __name__ == '__main__':
    main()
