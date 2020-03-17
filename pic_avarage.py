import cv2
import numpy
from matplotlib import pyplot as plt


# 读图
def read_pic():
    img = cv2.imread('均衡化处理.png')
    b, g, r = cv2.split(img)
    return b, g, r


# 得到通道的像素的分布字典
# rgb为图像通道
def get_rgb_dic(rgb):
    rgb_dic = {}
    data = []
    count = rgb.shape[0] * rgb.shape[1]
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
            if rgb[i][j] not in rgb_dic.keys():
                rgb_dic[rgb[i][j]] = 1
            else:
                rgb_dic[rgb[i][j]] += 1
    g_max = max(data)
    g_min = 0
    for i in range(g_max):
        if i not in rgb_dic.keys():
            rgb_dic[i] = 0
    return rgb_dic, count, g_max, g_min


# 根据像素分布频数字典得到像素的频率分布字典
# count为像素值个数总和
def get_histogram(rgb_dic, count):
    for key in rgb_dic.keys():
        rgb_dic[key] = rgb_dic[key] / count
    rgb_dic = sorted(rgb_dic.items(), key=lambda x: x[0], reverse=False)
    return rgb_dic


# 得到累积分布函数
def get_cumulative_distribution(rgb_dic):
    count = 0
    for key in rgb_dic.keys():
        count += rgb_dic[key]
        rgb_dic[key] = count
    return rgb_dic


# 为图像通道进行均衡化处理
def equalization(rgb):
    b_dic, b_count, g_max, g_min = get_rgb_dic(rgb)
    b_histogram_dic = dict(get_histogram(b_dic, b_count))
    cumulative_distribution_dic = get_cumulative_distribution(b_histogram_dic)
    diff = g_max - g_min
    rgb_g = []
    for key in cumulative_distribution_dic.keys():
        rgb_g.append(int(diff*cumulative_distribution_dic[key] + g_min + 0.5))
    return rgb_g


# 得到新的图像的直方图
def get_new_histogram(rgb_g, rgb):
    new_matrix = numpy.zeros((rgb.shape[0], rgb.shape[1]))
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            new_matrix[i][j] = rgb_g[rgb[i][j]-1]
    new_matrix = new_matrix.astype(numpy.uint8)
    return new_matrix


#根据像素分布画出直方图
def draw_pic(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


if __name__ == '__main__':
    b, g, r = read_pic()
    b_g = equalization(b)
    g_g = equalization(g)
    r_g = equalization(r)
    new_b_matrix = get_new_histogram(b_g, b)
    new_g_matrix = get_new_histogram(g_g, g)
    new_r_matrix = get_new_histogram(r_g, r)
    # draw_pic(new_b_matrix)
    cv2.imshow('pic', cv2.merge([new_b_matrix, new_g_matrix, new_r_matrix]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
