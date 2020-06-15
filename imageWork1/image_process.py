import cv2
import numpy
from matplotlib import pyplot as plt
import random
import math


def read_pic():
    img = cv2.imread('imageProcessing/pic.png')
    b, g, r = cv2.split(img)
    return b, g, r, img


def draw_gray_histogram(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


def draw_statistical_histogram(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, cumulative=True, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


def get_gray_binary_pic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray, binary


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


# 高斯滤波，rgb为图像通道, gaussian_kernel为高斯滤波的高斯核矩阵, value为矩阵长
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


# 双边滤波,rgb为图像通道, gaussian_kernel为高斯滤波的高斯核矩阵, value为矩阵长
# 双边滤波需要很大的sigma,大于150才有明显效果
def bilateral_filtering(rgb, gaussian_kernel, value, sigma, sigma_d):
    padding = int((gaussian_kernel.shape[0] - 1)/2)
    for i in range(padding, rgb.shape[0]-padding):
        for j in range(padding, rgb.shape[1]-padding):
            square_sum = 0
            bilateral_kernel = numpy.zeros((value, value))
            _count = 0
            for m in range(-padding, padding + 1):
                for n in range(-padding, padding + 1):
                    bilateral_kernel[m+padding][n+padding] = gaussian_kernel[m+padding][n+padding] * pow(
                        math.e, (-(math.fabs(rgb[i+m][j+n] - rgb[i][j])**2)/(2*pow(sigma_d, 2))))
                    _count += bilateral_kernel[m+padding][n+padding]
            for p in range(bilateral_kernel.shape[0]):
                for q in range(bilateral_kernel.shape[1]):
                    bilateral_kernel[p][q] = bilateral_kernel[p][q] / _count
            for m in range(-padding, padding + 1):
                for n in range(-padding, padding + 1):
                    square_sum += rgb[i+m][j+n] * \
                        bilateral_kernel[m+padding][n+padding]
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


# 为每个通道进行双边滤波处理
def integration_bilateral_filtering(b, g, r, value, sigma, sigma_d):
    gaussian_kernel = get_gaussian_kernel(value, sigma)
    padding_b, padding_g, padding_r = get_padding_rgb(b, g, r, gaussian_kernel)
    end_b_rgb = bilateral_filtering(
        padding_b, gaussian_kernel, value, sigma, sigma_d)
    end_g_rgb = bilateral_filtering(
        padding_g, gaussian_kernel, value, sigma, sigma_d)
    end_r_rgb = bilateral_filtering(
        padding_r, gaussian_kernel, value, sigma, sigma_d)
    return end_b_rgb, end_g_rgb, end_r_rgb


def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V


def rgb2hls(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    l = (mx + mn)/2
    H = h / 2
    S = s * 255.0
    L = l * 255.0
    return H, S, L


def get_hsv(img):
    b, g, r = cv2.split(img)
    H = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    S = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    V = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            H[i][j], S[i][j], V[i][j] = rgb2hsv(r[i][j], g[i][j], b[i][j])
    return H, S, V


def get_hls(img):
    b, g, r = cv2.split(img)
    H = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    S = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    L = numpy.zeros((b.shape[0], b.shape[1]), numpy.float32)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            H[i][j], S[i][j], L[i][j] = rgb2hsv(r[i][j], g[i][j], b[i][j])
    return H, S, L


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
    b, g, r, img = read_pic()
    new_img = img
    # img = cv2.boxFilter(new_img, -1, (2,2), normalize=0)
    # draw_gray_histogram(b)
    # draw_statistical_histogram(b)
    # new_img = sp_noise(img, 0.01)
    new_img = gasuss_noise(img, 0, 0.01)
    b, g, r = cv2.split(new_img)
    # gray, binary = get_gray_binary_pic(img)
    # cv2.imshow('pic', gray)
    value = 3
    sigma = 1.5
    sigma_d = 300
    end_b_rgb, end_g_rgb, end_r_rgb = integration_bilateral_filtering(
        b, g, r, value, sigma, sigma_d)
    draw_pic(end_b_rgb)
    # H, S, V = get_hsv(img)
    # cv2.imshow('pic', cv2.merge([H, S, V]))
    cv2.imshow('pic', cv2.merge([end_b_rgb, end_g_rgb, end_r_rgb]))
    # H,S,L = get_hls(img)
    # cv2.imshow('pic', cv2.merge([H, S, L]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
