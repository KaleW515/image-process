import cv2
import numpy
from matplotlib import pyplot as plt
import math
from skimage import morphology, data, color


def read_pic():
    img = cv2.imread('/home/kale/PycharmProjects/LogoDetection/HFUT-VL2/JPEGImages/1_20.jpg')
    b, g, r = cv2.split(img)
    return b, g, r, img


# 得到二值化
def get_gray_binary_pic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray, binary


# 按照像素值的分布画出灰度直方图
def draw_pic(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


def dilation(rgb, value):
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
            end_rgb[i][j] = area_sum[len(area_sum)-1]
    return end_rgb


# 图像膨胀操作
def integration_dilation(b, g, r, value):
    end_b_rgb = dilation(b, value)
    end_g_rgb = dilation(g, value)
    end_r_rgb = dilation(r, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


def erosion(rgb, value):
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
            end_rgb[i][j] = area_sum[0]
    return end_rgb


# 图像腐蚀操作
def integration_erosion(b, g, r, value):
    end_b_rgb = erosion(b, value)
    end_g_rgb = erosion(g, value)
    end_r_rgb = erosion(r, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 图像开操作
def pic_open(b, g, r, value):
    erosion_b_rgb, erosion_g_rgb, erosion_r_rgb = integration_erosion(
        b, g, r, value)
    end_b_rgb, end_g_rgb, end_r_rgb = integration_dilation(
        erosion_b_rgb, erosion_g_rgb, erosion_r_rgb, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 图像闭操作
def pic_close(b, g, r, value):
    dilation_b_rgb, dilation_g_rgb, dilation_r_rgb = integration_dilation(
        b, g, r, value)
    end_b_rgb, end_g_rgb, end_r_rgb = integration_erosion(
        dilation_b_rgb, dilation_g_rgb, dilation_r_rgb, value)
    return end_b_rgb, end_g_rgb, end_r_rgb


# 图像细化操作
def thinning(img):
    gray, binary = get_gray_binary_pic(img)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            binary[i][j] = 1 if binary[i][j] == 255 else 0
    new_binary = numpy.zeros(binary.shape, numpy.uint8)
    binary = cv2.copyMakeBorder(
        binary, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    temp_binary = cv2.copyMakeBorder(
        new_binary, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0,
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    for i in range(1, new_binary.shape[0]):
        for j in range(1, new_binary.shape[1]):
            sum_num = binary[i-1][j-1] + binary[i-1][j]*2 + binary[i-1][j+1]*4 + binary[i][j-1]*8 + \
                binary[i][j+1] * 16 + binary[i+1][j-1] * 32 + \
                binary[i+1][j] * 64 + binary[i+1][j+1]*128
            temp_binary[i][j] = 255 if array[sum_num] == 1 else 0
    for i in range(new_binary.shape[0]):
        for j in range(new_binary.shape[1]):
            new_binary[i][j] = temp_binary[i+1][j+1]
    return new_binary


def get_zhang_list(binary, i, j):
    temp_list = []
    temp_list.append(binary[i-1][j])
    temp_list.append(binary[i-1][j+1])
    temp_list.append(binary[i][j+1])
    temp_list.append(binary[i+1][j+1])
    temp_list.append(binary[i+1][j])
    temp_list.append(binary[i+1][j-1])
    temp_list.append(binary[i][j-1])
    temp_list.append(binary[i-1][j-1])
    return temp_list


def zhang_thinning(img):
    gray, binary = get_gray_binary_pic(img)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            binary[i][j] = 1 if binary[i][j] == 255 else 0
    new_binary = numpy.zeros(binary.shape, numpy.uint8)
    binary = cv2.copyMakeBorder(
        binary, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    mark = [0]
    mark = []
    for count in range(2):
        for i in range(1, new_binary.shape[0]):
            for j in range(1, new_binary.shape[1]):
                N = binary[i-1][j-1] + binary[i-1][j] + binary[i-1][j+1] + binary[i][j-1] + \
                    binary[i][j+1] + binary[i+1][j-1] + \
                    binary[i+1][j] + binary[i+1][j+1]
                temp_list = get_zhang_list(binary, i, j)
                S = 0
                for k in range(len(temp_list)-1):
                    if temp_list[k] == 0 and temp_list[k+1] == 1:
                        S += 1
                product_2_4_6 = temp_list[0]*temp_list[2]*temp_list[4]
                product_4_6_8 = temp_list[2]*temp_list[4]*temp_list[6]
                if (N >= 2) and (N <= 6) and (S == 1) and (product_2_4_6 == 0) and (product_4_6_8 == 0):
                    mark.append((i, j))
        for i in range(len(mark)):
            binary[mark[i][0]][mark[i][1]] = 1

        mark = []
        for i in range(1, new_binary.shape[0]):
            for j in range(1, new_binary.shape[1]):
                N = binary[i-1][j-1] + binary[i-1][j] + binary[i-1][j+1] + binary[i][j-1] + \
                    binary[i][j+1] + binary[i+1][j-1] + \
                    binary[i+1][j] + binary[i+1][j+1]
                temp_list = get_zhang_list(binary, i, j)
                S = 0
                for k in range(len(temp_list)-1):
                    if temp_list[k] == 0 and temp_list[k+1] == 1:
                        S += 1
                product_2_4_8 = temp_list[0]*temp_list[2]*temp_list[6]
                product_2_6_8 = temp_list[0]*temp_list[4]*temp_list[6]
                if (N >= 2) and (N <= 6) and (S == 1) and (product_2_4_8 == 0) and (product_2_6_8 == 0):
                    mark.append((i, j))
        for i in range(len(mark)):
            binary[mark[i][0]][mark[i][1]] = 1
    for i in range(new_binary.shape[0]):
        for j in range(new_binary.shape[1]):
            new_binary[i][j] = binary[i+1][j+1]
            new_binary[i][j] = 255 if new_binary[i][j] == 1 else 0
    return new_binary


if __name__ == '__main__':
    b, g, r, img = read_pic()
    value = 7
    # end_b_rgb, end_g_rgb, end_r_rgb = pic_open(b, g, r, value)
    # draw_pic(end_b_rgb)
    cv2.imshow('pic', cv2.merge([end_b_rgb, end_g_rgb, end_r_rgb]))
    # binary = thinning(img)
    # binary = zhang_thinning(img)
    # cv2.imshow('pic', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
