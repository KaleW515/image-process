import cv2
import numpy


def circular_LBP(img, radius=4, neighbors=8):
    """
    ~圆形LBP~
    rx, ry为采样点对于中心点坐标的偏移量
    x1,x2,y1,y2是为双线性插值做准备
    tx,ty为通过减运算将坐标偏移量映射到0到1之间
    w1,w2,w3,w4为根据权重计算公式计算的结果
    LBP特征图像的邻域的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
    """
    dst = numpy.zeros(
        (img.shape[0]-2*radius, img.shape[1]-2*radius), dtype=img.dtype)
    for k in range(neighbors):
        rx = radius * numpy.cos(2.0 * numpy.pi * k / neighbors)
        ry = -(radius * numpy.sin(2.0 * numpy.pi * k / neighbors))
        x1 = int(numpy.floor(rx))
        x2 = int(numpy.ceil(rx))
        y1 = int(numpy.floor(ry))
        y2 = int(numpy.ceil(ry))
        tx = rx - x1
        ty = ry - y1
        w1 = (1-tx) * (1-ty)
        w2 = tx * (1-ty)
        w3 = (1-tx) * ty
        w4 = tx * ty
        for i in range(radius, img.shape[0]-radius):
            for j in range(radius, img.shape[1]-radius):
                center = img[i, j]
                neighbor = img[i+y1, j+x1] * w1 + img[i+y2, j+x1] * \
                    w2 + img[i+y1, j+x2] * w3 + img[i+y2, j+x2] * w4
                dst[i-radius, j-radius] |= (neighbor >
                                            center) << (numpy.uint8)(neighbors-k-1)
    return dst


def rotation_invariant_LBP(img, radius=6, neighbors=8):
    """
    ~旋转不变LBP~
    rx, ry为采样点对于中心点坐标的偏移量
    x1,x2,y1,y2是为双线性插值做准备
    tx,ty为通过减运算将坐标偏移量映射到0到1之间
    w1,w2,w3,w4为根据权重计算公式计算的结果
    LBP特征图像的邻域的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
    旋转不变计算各种情况,取最小值
    """
    dst = numpy.zeros(
        (img.shape[0]-2*radius, img.shape[1]-2*radius), dtype=img.dtype)
    for k in range(neighbors):
        rx = radius * numpy.cos(2.0 * numpy.pi * k / neighbors)
        ry = -(radius * numpy.sin(2.0 * numpy.pi * k / neighbors))
        x1 = int(numpy.floor(rx))
        x2 = int(numpy.ceil(rx))
        y1 = int(numpy.floor(ry))
        y2 = int(numpy.ceil(ry))
        tx = rx - x1
        ty = ry - y1
        w1 = (1-tx) * (1-ty)
        w2 = tx * (1-ty)
        w3 = (1-tx) * ty
        w4 = tx * ty
        for i in range(radius, img.shape[0]-radius):
            for j in range(radius, img.shape[1]-radius):
                center = img[i, j]
                neighbor = img[i+y1, j+x1] * w1 + img[i+y2, j+x1] * \
                    w2 + img[i+y1, j+x2] * w3 + img[i+y2, j+x2] * w4
                dst[i-radius, j-radius] |= (neighbor >
                                            center) << (numpy.uint8)(neighbors-k-1)
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            currentValue = dst[i, j]
            minValue = currentValue
            for k in range(1, neighbors):
                temp = (numpy.uint8)(currentValue >> (neighbors-k)
                                     ) | (numpy.uint8)(currentValue << k)
                if temp < minValue:
                    minValue = temp
            dst[i, j] = minValue
    return dst


if __name__ == '__main__':
    img = cv2.imread('imageWork3/240be777a522347299bb9c221a077d42.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = circular_LBP(gray)
    cv2.imshow('pic', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
