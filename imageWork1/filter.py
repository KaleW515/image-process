import cv2
import numpy
import random
from matplotlib import pyplot as plt
import math


class Filter:
    __rgb = ""
    __img = ""

    def set_rgb(self, _rgb: numpy.ndarray):
        self.__rgb = _rgb

    def set_img(self, _img: numpy.ndarray):
        self.__img = _img

    # 添加椒盐噪音
    # prob是阈值
    @staticmethod
    def __sp_noise(prob, rgb):
        sp_rgb = numpy.zeros(rgb.shape, numpy.uint8)
        threshold = 1 - prob
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                tmp = random.random()
                if tmp < prob:
                    sp_rgb[i][j] = 0
                elif tmp > threshold:
                    sp_rgb[i][j] = 255
                else:
                    sp_rgb[i][j] = rgb[i][j]
        return sp_rgb

    # 对每个通道进行椒盐噪声的添加
    def integration_sp_noise(self, prob):
        b, g, r = cv2.split(self.__img)
        res_img = cv2.merge([self.__sp_noise(prob, b), self.__sp_noise(prob, g), self.__sp_noise(prob, r)])
        return res_img

    # 添加高斯噪声
    # mean是均值, variance是方差
    @staticmethod
    def __gasuss_noise(rgb, mean, variance):
        rgb = numpy.array(rgb / 255, dtype=float)
        noise = numpy.random.normal(mean, variance ** 0.5, rgb.shape)
        gasuss_img = rgb + noise
        if gasuss_img.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        gasuss_img = numpy.clip(gasuss_img, low_clip, 1.0)
        gasuss_img = numpy.uint8(gasuss_img * 255)
        return gasuss_img

    # 对每个通道进行高斯噪声的添加
    def integration_gasuss_noise(self, mean, variance):
        b, g, r = cv2.split(self.__img)
        res_img = cv2.merge([self.__gasuss_noise(b, mean, variance), self.__gasuss_noise(g, mean, variance),
                             self.__gasuss_noise(r, mean, variance)])
        return res_img

    # 均值滤波, rgb为图像通道, value为矩阵长
    @staticmethod
    def __average_filtering(rgb, value):
        border_num = int((value - 1) / 2)
        end_rgb = numpy.zeros(rgb.shape, numpy.uint8)
        wide = rgb.shape[0]
        height = rgb.shape[1]
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                count = 0  # 记录周围可用块数量
                area_sum = 0
                for m in range(-border_num, border_num + 1):
                    for n in range(-border_num, border_num + 1):
                        if ((i + m) >= 0) & ((j + n) >= 0) & ((i + m) <= wide - 1) & ((j + n) <= height - 1):
                            count += 1
                            area_sum += rgb[i + m][j + n]
                end_rgb[i][j] = int(area_sum / count)
        return end_rgb

    # 为每个通道进行均值滤波处理
    def integration_average_filtering(self, value):
        b, g, r = cv2.split(self.__img)
        return self.__average_filtering(b, value), self.__average_filtering(g, value), self.__average_filtering(r,
                                                                                                                value)

    # 中值滤波, rgb为图像通道, value为矩阵长
    @staticmethod
    def __medium_filtering(rgb, value):
        border_num = int((value - 1) / 2)
        end_rgb = numpy.zeros(rgb.shape, numpy.uint8)
        wide = rgb.shape[0]
        height = rgb.shape[1]
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                area_sum = []
                for m in range(-border_num, border_num + 1):
                    for n in range(-border_num, border_num + 1):
                        if ((i + m) >= 0) & ((j + n) >= 0) & ((i + m) <= wide - 1) & ((j + n) <= height - 1):
                            area_sum.append(rgb[i + m][j + n])
                area_sum = sorted(area_sum)
                end_rgb[i][j] = area_sum[len(area_sum) // 2]
        return end_rgb

    # 为每个通道进行中值滤波处理
    def integration_median_filtering(self, value):
        b, g, r = cv2.split(self.__img)
        return self.__medium_filtering(b, value), self.__medium_filtering(g, value), self.__medium_filtering(r,
                                                                                                             value)

    # 得到高斯滤波的高斯核矩阵, value是矩阵长, sigma为方差
    @staticmethod
    def __get_gaussian_kernel(value, sigma):
        border_num = int((value - 1) / 2)
        kernel_matrix = numpy.zeros((value, value))
        count = 0
        for i in range(-border_num, border_num + 1):
            for j in range(-border_num, border_num + 1):
                kernel_matrix[i + border_num][j + border_num] = (1 / (2 * math.pi * pow(sigma, 2))) * pow(math.e, (
                        -(i ** 2 + j ** 2) / (2 * pow(sigma, 2))))
                count += kernel_matrix[i + border_num][j + border_num]
        _count = 0
        for i in range(kernel_matrix.shape[0]):
            for j in range(kernel_matrix.shape[1]):
                kernel_matrix[i][j] = kernel_matrix[i][j] / count
                _count += kernel_matrix[i][j]
        return kernel_matrix

    # 根据矩阵长为原始图像的每个通道进行边缘扩充处理
    # 处理方法是将边缘的像素向外进行扩展
    def __get_padding_rgb(self, gaussian_kernel):
        b, g, r = cv2.split(self.__img)
        padding = int((gaussian_kernel.shape[0] - 1) / 2)
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
    @staticmethod
    def __gaussian_filtering(rgb, gaussian_kernel, value):
        padding = int((gaussian_kernel.shape[0] - 1) / 2)
        for i in range(padding, rgb.shape[0] - padding):
            for j in range(padding, rgb.shape[1] - padding):
                square_sum = 0
                for m in range(-padding, padding + 1):
                    for n in range(-padding, padding + 1):
                        square_sum += rgb[i + m][j + n] * \
                                      gaussian_kernel[m + padding][n + padding]
                rgb[i][j] = square_sum
        end_rgb = numpy.zeros((rgb.shape[0] - 2 * padding, rgb.shape[1] - 2 * padding))
        for i in range(end_rgb.shape[0]):
            for j in range(end_rgb.shape[1]):
                end_rgb[i][j] = rgb[i + padding][j + padding]
        end_rgb = end_rgb.astype(numpy.uint8)
        return end_rgb

    # 为每个通道进行高斯滤波处理
    def integration_gaussian_filtering(self, value, sigma):
        gaussian_kernel = self.__get_gaussian_kernel(value, sigma)
        padding_b, padding_g, padding_r = self.__get_padding_rgb(gaussian_kernel)
        return self.__gaussian_filtering(padding_b, gaussian_kernel, value), self.__gaussian_filtering(padding_g,
                                                                                                       gaussian_kernel,
                                                                                                       value), \
               self.__gaussian_filtering(
                   padding_r, gaussian_kernel, value)


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
    f = Filter()
    f.set_img(cv2.imread("pic.png"))
    img = cv2.merge([f.integration_sp_noise(0.001)])
    cv2.imshow("pic", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
