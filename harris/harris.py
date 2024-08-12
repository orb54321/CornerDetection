import cv2
import numpy as np

def HarrisCornerDetection(input_img, window_size, k, threshold, sigma):
    corner_list = []
    output_img = input_img
    offset = int(window_size / 2)

    img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2GRAY)

    y_range = img.shape[0] - offset
    x_range = img.shape[1] - offset

    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    gaussian_weight = np.exp(-(i**2 + j**2) / (2 * sigma**2))

    for y in range(offset, y_range):
        for x in range(offset, x_range):
            #sliding window
            start_x = x - offset
            end_x = x + offset + 1
            start_y = y - offset
            end_y = y + offset + 1

            window_Ixx = Ixx[start_y:end_y, start_x:end_x] * gaussian_weight
            window_Ixy = Ixy[start_y:end_y, start_x:end_x] * gaussian_weight
            window_Iyy = Iyy[start_y:end_y, start_x:end_x] * gaussian_weight

            Sxx = window_Ixx.sum()
            Sxy = window_Ixy.sum()
            Syy = window_Iyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            r = det - k * (trace**2)
            if r > threshold:
                corner_list.append([x, y, r])
                output_img[y, x] = (0, 0, 255)
    return corner_list, output_img

def main():
    window_size = 5
    k = 0.04
    threshold = 10000000.0
    sigma = 10
    input_img = cv2.imread('input3.jpg')
    input_img = cv2.resize(input_img, (0, 0), fx=0.5, fy=0.5)       #if the image is too big, resize it
    cv2.imshow('input', input_img)
    cv2.waitKey(0)
    if input_img is not None:
        print('Harris corner detection start...')
        corner_list, output_img = HarrisCornerDetection(input_img, window_size, k, threshold, sigma)
        if output_img is not None:
            cv2.imwrite('harris_corner.jpg', output_img)
        cv2.imshow('Harris corner detection', output_img)
        cv2.waitKey(0)
        print('Harris corner detection end...')
    else:
        print('Image not found')

if __name__ == '__main__':
    main()