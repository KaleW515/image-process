import cv2
import numpy
import random
from matplotlib import pyplot as plt
import math


# 添加椒盐噪音
# prob是阈值
def sp_noise(img, prob):
    new_img = numpy.zeros(img.shape, numpy.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            num = random.random()
            if num < prob:
                new_img[i][j] = 0
            elif num > thres:
                new_img[i][j] = 255
            else:
                new_img[i][j] = img[i][j]
    return new_img


# 对每个通道进行椒盐噪声的添加
def integration_sp_noise(img, prob):
    b, g, r = cv2.split(img)
    new_b = sp_noise(b, prob)
    new_g = sp_noise(g, prob)
    new_r = sp_noise(r, prob)
    new_img = cv2.merge([new_b, new_g, new_r])
    return new_img


# 添加高斯噪声
# mean是均值, variance是方差
def gasuss_noise(img, mean, variance):
    img = numpy.array(img/255, dtype=float)
    noise = numpy.random.normal(mean, variance ** 0.5, img.shape)
    new_img = img + noise
    if new_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    new_img = numpy.clip(new_img, low_clip, 1.0)
    new_img = numpy.uint8(new_img*255)
    return new_img


# 对每个通道进行高斯噪声的添加
def integration_gasuss_noise(img, mean, variance):
    b, g, r = cv2.split(img)
    new_b = gasuss_noise(b, mean, variance)
    new_g = gasuss_noise(g, mean, variance)
    new_r = gasuss_noise(r, mean, variance)
    new_img = cv2.merge([new_b, new_g, new_r])
    return new_img


# 均值滤波, rgb为图像通道, value为矩阵长
def average_filtering(rgb, value):
    border_num = int((value-1)/2)
    end_rgb = numpy.zeros(rgb.shape, numpy.uint8)
    wide = rgb.shape[0]
    height = rgb.shape[1]
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            count = 0  # 记录周围可用块数量
            area_sum = 0
            for m in range(-border_num, border_num+1):
                for n in range(-border_num, border_num+1):
                    if ((i+m) >= 0) & ((j+n) >= 0) & ((i+m) <= wide-1) & ((j+n) <= height-1):
                        count += 1
                        area_sum += rgb[i+m][j+n]
            end_rgb[i][j] = int(area_sum/count)
    return end_rgb


# 中值滤波, rgb为图像通道, value为矩阵长
def median_filtering(rgb, value):
    border_num = int((value-1)/2)
    end_rgb = numpy.zeros(rgb.shape, numpy.uint8)
    wide = rgb.shape[0]
    height = rgb.shape[1]
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            area_sum = []
            for m in range(-border_num, border_num+1):
                for n in range(-border_num, border_num+1):
                    if ((i+m) >= 0) & ((j+n) >= 0) & ((i+m) <= wide-1) & ((j+n) <= height-1):
                        area_sum.append(rgb[i+m][j+n])
            area_sum = sorted(area_sum)
            end_rgb[i][j] = area_sum[len(area_sum)//2]
    return end_rgb


# 得到高斯滤波的高斯核矩阵, value是矩阵长, sigma为方差
def get_gaussian_kernel(value, sigma):
    border_num = int((value - 1)/2)
    kernel_matrix = numpy.zeros((value, value))
    count = 0
    for i in range(-border_num, border_num+1):
        for j in range(-border_num, border_num+1):
            kernel_matrix[i+border_num][j+border_num] = (
                1/(2*math.pi*pow(sigma, 2)))*pow(math.e, (-(i**2+j**2)/(2*pow(sigma, 2))))
            count += kernel_matrix[i+border_num][j+border_num]
    _count = 0
    for i in range(kernel_matrix.shape[0]):
        for j in range(kernel_matrix.shape[1]):
            kernel_matrix[i][j] = kernel_matrix[i][j] / count
            _count += kernel_matrix[i][j]
    return kernel_matrix


# 根据矩阵长为原始图像的每个通道进行边缘扩充处理
# 处理方法是将边缘的像素向外进行扩展
def get_padding_rgb(b, g, r, gaussian_kernel):
    padding = int((gaussian_kernel.shape[0] - 1)/2)
    padding_b = cv2.copyMakeBorder(
        b, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    padding_b = padding_b.astype(numpy.uint8)
    padding_g = cv2.copyMakeBorder(
        g, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    padding_g = padding_g.astype(numpy.uint8)
    padding_r = cv2.copyMakeBorder(
        r, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    padding_r = padding_r.astype(numpy.uint8)
    return padding_b, padding_g, padding_r


 # 高斯滤波，　rgb为图像通道, gaussian_kernel为高斯滤波的高斯核矩阵, value为矩阵长
def gaussian_filtering(rgb, gaussian_kernel, value):
    padding = int((gaussian_kernel.shape[0] - 1)/2)
    for i in range(padding, rgb.shape[0]-padding):
        for j in range(padding, rgb.shape[1]-padding):
            square_sum = 0
            for m in range(-padding, padding + 1):
                for n in range(-padding, padding + 1):
                    square_sum += rgb[i+m][j+n] * \
                        gaussian_kernel[m+padding][n+padding]
            rgb[i][j] = square_sum
    end_rgb = numpy.zeros((rgb.shape[0]-2*padding, rgb.shape[1]-2*padding))
    for i in range(end_rgb.shape[0]):
        for j in range(end_rgb.shape[1]):
            end_rgb[i][j] = rgb[i+padding][j+padding]
    end_rgb = end_rgb.astype(numpy.uint8)
    return end_rgb


# 为每个通道进行均值滤波处理
def integration_average_filtering(b, g, r, value):
    end_b_rgb = average_filtering(b, value)
    end_g_rgb = average_filtering(g, value)
    end_r_rgb = average_filtering(r, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 为每个通道进行中值滤波处理
def integration_median_filtering(b, g, r, value):
    end_b_rgb = median_filtering(b, value)
    end_g_rgb = median_filtering(g, value)
    end_r_rgb = median_filtering(r, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 为每个通道进行高斯滤波处理
def integration_gaussian_filtering(b, g, r, value, sigma):
    gaussian_kernel = get_gaussian_kernel(value, sigma)
    padding_b, padding_g, padding_r = get_padding_rgb(b, g, r, gaussian_kernel)
    end_b_rgb = gaussian_filtering(padding_b, gaussian_kernel, value)
    end_g_rgb = gaussian_filtering(padding_g, gaussian_kernel, value)
    end_r_rgb = gaussian_filtering(padding_r, gaussian_kernel, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 按照像素值的分布画出灰度直方图
def draw_pic(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('imageProcessing/pic.png')
    new_img = img
    # new_img = sp_noise(img, 0.001)
    new_img = gasuss_noise(img, 0, 0.1)
    b, g, r = cv2.split(new_img)
    # cv2.imshow('pic', new_img)
    # draw_pic(b)
    value = 3
    sigma = 1.5
    # end_b_rgb, end_g_rgb, end_r_rgb = integration_average_filtering(
    #     b, g, r, value)
    end_b_rgb, end_g_rgb, end_r_rgb = integration_median_filtering(
        b, g, r, value)
    # end_b_rgb, end_g_rgb, end_r_rgb = integration_gaussian_filtering(
    #     b, g, r, value, sigma)
    draw_pic(end_b_rgb)
    # cv2.imshow('pic', cv2.merge([end_b_rgb, end_g_rgb, end_r_rgb]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
