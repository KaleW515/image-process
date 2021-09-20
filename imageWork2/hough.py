import numpy
import matplotlib.pyplot as plt
import cv2


class Hough:
    __img = ""

    def set_img(self, _img: numpy.ndarray):
        self.__img = _img

    @staticmethod
    def hough(_edge, ThetaDim, DistStep, threshold=None):
        max_dis = numpy.sqrt(_edge.shape[0] ** 2 + _edge.shape[1] ** 2)
        dis_dim = int(numpy.ceil(max_dis / DistStep))
        half_dist_window_size = int(dis_dim / 50)
        accumulator = numpy.zeros((ThetaDim, dis_dim))
        sin_theta = [numpy.sin(t * numpy.pi / ThetaDim) for t in range(ThetaDim)]
        cos_theta = [numpy.cos(t * numpy.pi / ThetaDim) for t in range(ThetaDim)]
        for i in range(_edge.shape[0]):
            for j in range(_edge.shape[1]):
                if not _edge[i, j] == 0:
                    for k in range(ThetaDim):
                        accumulator[k][int(
                            round((i * cos_theta[k] + j * sin_theta[k]) * dis_dim / max_dis))] += 1
        M = accumulator.max()
        if threshold is None:
            threshold = int(M / 5)
        result = numpy.array(numpy.where(accumulator > threshold))
        temp = [[], []]
        for i in range(result.shape[1]):
            eight_neighbour = accumulator[max(0, result[0, i] - 3):min(result[0, i] + 2, accumulator.shape[0]), max(
                0, result[1, i] - half_dist_window_size + 1):min(result[1, i] + half_dist_window_size,
                                                                 accumulator.shape[1])]
            if (accumulator[result[0, i], result[1, i]] >= eight_neighbour).all():
                temp[0].append(result[0, i])
                temp[1].append(result[1, i])
        result = numpy.array(temp)
        result = result.astype(numpy.float64)
        result[0] = result[0] * numpy.pi / ThetaDim
        result[1] = result[1] * max_dis / dis_dim
        return result


def draw_lines(lines, blurred, color=(255, 165, 0), err=3):
    result = blurred
    Cos = numpy.cos(lines[0])
    Sin = numpy.sin(lines[0])
    for i in range(blurred.shape[0]):
        for j in range(blurred.shape[1]):
            e = numpy.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any():
                result[i, j] = color
    return result


if __name__ == '__main__':
    blurred = cv2.GaussianBlur(plt.imread('IMG_20200401_171217.jpg'), (3, 3), 0)
    edge = cv2.Canny(blurred, 50, 150)
    hough = Hough()
    hough.set_img(plt.imread('IMG_20200401_171217.jpg'))
    final_img = draw_lines(hough.hough(edge, 90, 1), blurred)
    plt.imshow(final_img, cmap='gray')
    plt.axis('off')
    plt.show()
