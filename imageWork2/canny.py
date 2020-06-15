import cv2
import numpy


def sobel(blurred_rgb):
    sobel_kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    result = numpy.zeros(
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
            result[i - 1][j - 1] = numpy.sqrt(sum_num_x ** 2 + sum_num_y ** 2)
            theta[(i - 1, j - 1)] = (sum_num_x, sum_num_y)
    return result, theta


def nms(result, theta):
    result = cv2.copyMakeBorder(
        result, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    nms_result = numpy.zeros(result.shape, numpy.uint8)
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if theta[(i - 1, j - 1)][0] == 0:
                if result[i][j] > result[i - 1][j] and result[i][j] > result[i + 1][j]:
                    nms_result[i - 1][j - 1] = result[i][j]

            elif theta[(i - 1, j - 1)][1] == 0:
                if result[i][j] > result[i][j - 1] and result[i][j] > result[i][j + 1]:
                    nms_result[i - 1][j - 1] = result[i][j]

            else:
                if theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > 1:
                    weight = numpy.fabs(
                        theta[(i - 1, j - 1)][0]) / numpy.fabs(theta[(i - 1, j - 1)][1])
                    dTmp1 = weight * result[i - 1][j + 1] + \
                            (1 - weight) * result[i - 1][j]
                    dTmp2 = weight * result[i + 1][j - 1] + \
                            (1 - weight) * result[i + 1][j]
                    if result[i][j] > dTmp1 and result[i][j] > dTmp2:
                        nms_result[i - 1][j - 1] = result[i][j]

                elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < -1:
                    weight = numpy.fabs(
                        theta[(i - 1, j - 1)][0]) / numpy.fabs(theta[(i - 1, j - 1)][1])
                    dTmp1 = weight * result[i - 1][j - 1] + \
                            (1 - weight) * result[i - 1][j]
                    dTmp2 = weight * result[i + 1][j + 1] + \
                            (1 - weight) * result[i + 1][j]
                    if result[i][j] > dTmp1 and result[i][j] > dTmp2:
                        nms_result[i - 1][j - 1] = result[i][j]

                elif (theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > -1) and (
                        theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < 0):
                    weight = numpy.fabs(
                        theta[(i - 1, j - 1)][1]) / numpy.fabs(theta[(i - 1, j - 1)][0])
                    dTmp1 = weight * result[i - 1][j - 1] + \
                            (1 - weight) * result[i][j - 1]
                    dTmp2 = weight * result[i + 1][j + 1] + \
                            (1 - weight) * result[i][j + 1]
                    if (result[i][j] > dTmp1) and (result[i][j] > dTmp2):
                        nms_result[i - 1][j - 1] = result[i][j]

                elif (theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] < 1) and (
                        theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] > 0):
                    weight = numpy.fabs(
                        theta[(i - 1, j - 1)][1]) / numpy.fabs(theta[(i - 1, j - 1)][0])
                    dTmp1 = weight * result[i + 1][j - 1] + \
                            (1 - weight) * result[i][j - 1]
                    dTmp2 = weight * result[i - 1][j + 1] + \
                            (1 - weight) * result[i][j + 1]
                    if (result[i][j] > dTmp1) and (result[i][j] > dTmp2):
                        nms_result[i - 1][j - 1] = result[i][j]
                elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] == 1:
                    if result[i][j] > result[i - 1][j + 1] and result[i][j] > result[i + 1][j - 1]:
                        nms_result[i - 1][j - 1] = result[i][j]
                elif theta[(i - 1, j - 1)][1] / theta[(i - 1, j - 1)][0] == -1:
                    if result[i][j] > result[i - 1][j - 1] and result[i][j] > result[i + 1][j + 1]:
                        nms_result[i - 1][j - 1] = result[i][j]
    return nms_result


def threshold(result, low_value, high_value):
    mark = {}
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j] >= high_value:
                mark[(i, j)] = 2
            elif result[i][j] < high_value and result[i][j] >= low_value:
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


def get_padding_rgb(blurred):
    b, g, r = cv2.split(blurred)
    padding_b = cv2.copyMakeBorder(
        b, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_b = padding_b.astype(numpy.uint8)
    padding_g = cv2.copyMakeBorder(
        g, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_g = padding_g.astype(numpy.uint8)
    padding_r = cv2.copyMakeBorder(
        r, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    padding_r = padding_r.astype(numpy.uint8)
    return padding_b, padding_g, padding_r


def canny(img, low_value, high_value):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray, theta = sobel(gray)
    result = nms(gray, theta)
    result = threshold(result, low_value, high_value)
    return result


if __name__ == '__main__':
    img = cv2.imread('image/imageWork2/2756259253df5d0330befd854.jpg')
    result = canny(img, 50, 100)
    cv2.imshow('pic', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
