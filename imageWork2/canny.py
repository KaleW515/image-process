import cv2
import numpy


class Canny:
    __img = ""

    def set_img(self, _img):
        self.__img = _img

    @staticmethod
    def __sobel(blurred_rgb):
        sobel_kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        _result = numpy.zeros(
            (blurred_rgb.shape[0] - 2, blurred_rgb.shape[1] - 2), numpy.uint8)
        theta = {}
        for i in range(1, blurred_rgb.shape[0] - 1):
            for j in range(1, blurred_rgb.shape[1] - 1):
                sum_num_x = 0
                sum_num_y = 0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        sum_num_x += blurred_rgb[i + m][j +
                                                        n] * sobel_kernel_x[m + 1][n + 1]
                        sum_num_y += blurred_rgb[i + m][j +
                                                        n] * sobel_kernel_y[m + 1][n + 1]
                _result[i - 1][j - 1] = numpy.sqrt(sum_num_x ** 2 + sum_num_y ** 2)
                theta[(i - 1, j - 1)] = (sum_num_x, sum_num_y)
        return _result, theta

    @staticmethod
    def __nms(_result, theta):
        _result = cv2.copyMakeBorder(
            _result, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        nms_result = numpy.zeros(_result.shape, numpy.uint8)
        for i in range(1, _result.shape[0] - 1):
            for j in range(1, _result.shape[1] - 1):
                if theta[(i - 1, j - 1)][0] == 0:
                    if _result[i][j] > _result[i - 1][j] and _result[i][j] > _result[i + 1][j]:
                        nms_result[i - 1][j - 1] = _result[i][j]

                elif theta[(i - 1, j - 1)][1] == 0:
                    if _result[i][j] > _result[i][j - 1] and _result[i][j] > _result[i][j + 1]:
                        nms_result[i - 1][j - 1] = _result[i][j]

                else:
                    if theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > 1:
                        weight = numpy.fabs(
                            theta[(i - 1, j - 1)][0]) / numpy.fabs(theta[(i - 1, j - 1)][1])
                        dTmp1 = weight * _result[i - 1][j + 1] + \
                                (1 - weight) * _result[i - 1][j]
                        dTmp2 = weight * _result[i + 1][j - 1] + \
                                (1 - weight) * _result[i + 1][j]
                        if _result[i][j] > dTmp1 and _result[i][j] > dTmp2:
                            nms_result[i - 1][j - 1] = _result[i][j]

                    elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < -1:
                        weight = numpy.fabs(
                            theta[(i - 1, j - 1)][0]) / numpy.fabs(theta[(i - 1, j - 1)][1])
                        dTmp1 = weight * _result[i - 1][j - 1] + \
                                (1 - weight) * _result[i - 1][j]
                        dTmp2 = weight * _result[i + 1][j + 1] + \
                                (1 - weight) * _result[i + 1][j]
                        if _result[i][j] > dTmp1 and _result[i][j] > dTmp2:
                            nms_result[i - 1][j - 1] = _result[i][j]

                    elif (theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > -1) and (
                            theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < 0):
                        weight = numpy.fabs(
                            theta[(i - 1, j - 1)][1]) / numpy.fabs(theta[(i - 1, j - 1)][0])
                        dTmp1 = weight * _result[i - 1][j - 1] + \
                                (1 - weight) * _result[i][j - 1]
                        dTmp2 = weight * _result[i + 1][j + 1] + \
                                (1 - weight) * _result[i][j + 1]
                        if (_result[i][j] > dTmp1) and (_result[i][j] > dTmp2):
                            nms_result[i - 1][j - 1] = _result[i][j]

                    elif (theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < 1) and (
                            theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > 0):
                        weight = numpy.fabs(
                            theta[(i - 1, j - 1)][1]) / numpy.fabs(theta[(i - 1, j - 1)][0])
                        dTmp1 = weight * _result[i + 1][j - 1] + \
                                (1 - weight) * _result[i][j - 1]
                        dTmp2 = weight * _result[i - 1][j + 1] + \
                                (1 - weight) * _result[i][j + 1]
                        if (_result[i][j] > dTmp1) and (_result[i][j] > dTmp2):
                            nms_result[i - 1][j - 1] = _result[i][j]
                    elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] == 1:
                        if _result[i][j] > _result[i - 1][j + 1] and _result[i][j] > _result[i + 1][j - 1]:
                            nms_result[i - 1][j - 1] = _result[i][j]
                    elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] == -1:
                        if _result[i][j] > _result[i - 1][j - 1] and _result[i][j] > _result[i + 1][j + 1]:
                            nms_result[i - 1][j - 1] = _result[i][j]
        return nms_result

    @staticmethod
    def __threshold(result, low_value, high_value):
        mark = {}
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i][j] >= high_value:
                    mark[(i, j)] = 2
                elif high_value > result[i][j] >= low_value:
                    mark[(i, j)] = 1
                else:
                    mark[(i, j)] = 0
                    result[i][j] = 0
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if mark[(i, j)] == 2:
                    pass
                elif mark[(i, j)] == 1:
                    flag = 0
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            try:
                                if mark[(i + m, j + n)] == 2:
                                    flag = 1
                                    break
                                else:
                                    pass
                            except:
                                pass
                    if flag == 0:
                        result[i][j] = 0
                else:
                    result[i][j] = 0
        return result

    def canny(self, low_value, high_value):
        gray, theta = self.__sobel(cv2.cvtColor(cv2.GaussianBlur(self.__img, (3, 3), 0), cv2.COLOR_BGR2GRAY))
        return self.__threshold(self.__nms(gray, theta), low_value, high_value)


if __name__ == '__main__':
    canny = Canny()
    canny.set_img(cv2.imread('2756259253df5d0330befd854.jpg'))
    cv2.imshow('pic', canny.canny(50, 100))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
