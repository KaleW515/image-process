import cv2
import numpy
from matplotlib import pyplot as plt


class PicAverage:
    __rgb = ""

    def set_rgb(self, rgb: numpy.ndarray):
        self.__rgb = rgb

    # 得到通道的像素的分布字典
    def get_rgb_dic(self):
        rgb_dic = {}
        rgb_max = 0
        rgb_min = 0
        for i in range(self.__rgb.shape[0]):
            for j in range(self.__rgb.shape[1]):
                rgb_max = max(rgb_max, int(self.__rgb[i][j]))
                if self.__rgb[i][j] not in rgb_dic.keys():
                    rgb_dic[self.__rgb[i][j]] = 1
                else:
                    rgb_dic[self.__rgb[i][j]] += 1
        for i in range(rgb_max):
            if i not in rgb_dic.keys():
                rgb_dic[i] = 0
        return rgb_dic, rgb_max, rgb_min

    # 根据像素分布频数字典得到像素的频率分布字典
    def get_histogram(self, rgb_dic):
        for key in rgb_dic.keys():
            rgb_dic[key] = rgb_dic[key] / (self.__rgb.shape[0] * self.__rgb.shape[1])
        rgb_dic = sorted(rgb_dic.items(), key=lambda x: x[0], reverse=False)
        return rgb_dic

    # 为图像通道进行均衡化处理
    def equalization(self):
        rgb_dic, rgb_max, rgb_min = self.get_rgb_dic()
        rgb_histogram_dic = dict(self.get_histogram(rgb_dic))
        count = 0
        for key in rgb_histogram_dic.keys():
            count += rgb_histogram_dic[key]
            rgb_histogram_dic[key] = count
        diff = rgb_max - rgb_min
        rgb_g = []
        for key in rgb_histogram_dic.keys():
            rgb_g.append(int(diff * rgb_histogram_dic[key] + rgb_min + 0.5))
        return rgb_g

    # 得到新的图像的直方图
    def get_new_histogram(self, rgb_g):
        new_matrix = numpy.zeros((self.__rgb.shape[0], self.__rgb.shape[1]))
        for i in range(self.__rgb.shape[0]):
            for j in range(self.__rgb.shape[1]):
                new_matrix[i][j] = rgb_g[self.__rgb[i][j] - 1]
        new_matrix = new_matrix.astype(numpy.uint8)
        return new_matrix


# 根据像素分布画出直方图
def draw_pic(rgb):
    data = []
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            data.append(int(rgb[i][j]))
    plt.hist(data, bins=256, facecolor="blue",
             edgecolor="black", alpha=0.7)
    plt.show()


if __name__ == '__main__':
    b, g, r = cv2.split(cv2.imread("248978-106.jpg"))
    pic = PicAverage()
    pic.set_rgb(b)
    new_b_matrix = pic.get_new_histogram(pic.equalization())
    pic.set_rgb(g)
    new_g_matrix = pic.get_new_histogram(pic.equalization())
    pic.set_rgb(r)
    new_r_matrix = pic.get_new_histogram(pic.equalization())
    # draw_pic(new_b_matrix)
    cv2.imshow('pic', cv2.merge([new_b_matrix, new_g_matrix, new_r_matrix]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
