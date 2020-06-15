import cv2
import numpy
from skimage.feature import hog


def get_gradient_magnitude():
    """
    ~得到梯度幅值和角度~
    分别计算x轴卷积result_x和y轴卷积result_y
    根据公式计算最终梯度幅值result, 为了防止0值作为除数,所以直接以字典形式返回gx和gy,在需要用的时候再计算theta
    """
    img = cv2.imread('imageWork3/64_128.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = numpy.power(gray/255, 0.8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gray[i][j] *= 255
    gray = gray.astype(numpy.uint8)
    result_x = numpy.zeros(gray.shape)
    result_y = numpy.zeros(gray.shape)
    tan = {}
    for i in range(gray.shape[0]):
        for j in range(1, gray.shape[1]-1):
            g_x = int(gray[i][j+1]) - int(gray[i][j-1])
            result_x[i][j] = g_x
    for i in range(gray.shape[0]):
        g_x_0 = int(gray[i][1]) - int(gray[i][0])
        g_x_len = int(gray[i][gray.shape[1]-1]) - int(gray[i][gray.shape[1]-2])
        result_x[i][0] = g_x_0
        result_x[i][gray.shape[1]-1] = g_x_len
    for i in range(1, gray.shape[0]-1):
        for j in range(gray.shape[1]):
            g_y = int(gray[i+1][j]) - int(gray[i-1][j])
            result_y[i][j] = g_y
    for j in range(gray.shape[1]):
        g_y_0 = int(gray[1][j]) - int(gray[0][j])
        g_y_len = int(gray[gray.shape[0]-1][j]) - int(gray[gray.shape[0]-2][j])
        result_y[0][j] = g_x_0
        result_y[gray.shape[0]-1][j] = g_y_len
    result = numpy.zeros(gray.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = numpy.sqrt(result_x[i][j]**2 + result_y[i][j]**2)
            tan[(i, j)] = (result_x[i][j], result_y[i][j])
    return result, tan


def get_bin(result, tan, dis=9):
    """
    ~得到bin,默认分成9份~
    对整幅图像进行bin的计算
    """
    bin_dic = {}
    bin_matrix = numpy.zeros(result.shape)
    step = 360/dis/2
    for i in range(1, dis+1):
        bin_dic[i] = [(i*step-step, i*step), (i*step-step-180, i*step-180)]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if tan[(i, j)][0] == 0:
                bin_matrix[i][j] = (dis+1)/2
            else:
                temp = 360*numpy.arctan(tan[(i, j)]
                                        [1] / tan[(i, j)][0])/numpy.pi
                for key in bin_dic.keys():
                    if (temp < bin_dic[key][0][1] and temp >= bin_dic[key][0][0]) or (temp < bin_dic[key][1][1] and temp >= bin_dic[key][1][0]):
                        bin_matrix[i][j] = int(key)
    return bin_matrix


def get_hog(dis):
    """
    ~得到hog特征~
    将图像先分为8*8的cell, v_cell为每个cell的特征,字典格式
    将cell按2*2的格式组合成block,不断滑动,得到v_block, 并进行归一化处理
    将所有的block的特征值串联起来,得到最终的hog特征
    """
    result, tan = get_gradient_magnitude()
    bin_matrix = get_bin(result, tan)
    v_cell = {}
    for i in range(0, result.shape[0], 8):
        for j in range(0, result.shape[1], 8):
            v_cell[(i, j)] = {}
            for m in range(8):
                for n in range(8):
                    if bin_matrix[i+m][j+n] not in v_cell[(i, j)].keys():
                        v_cell[(i, j)][int(bin_matrix[i+m][j+n])
                                       ] = int(result[i+m][j+n])
                    else:
                        v_cell[(i, j)][int(bin_matrix[i+m][j+n])
                                       ] += int(result[i+m][j+n])
            # print(v_cell[(i, j)])
    v_block = {}
    for i in range(0, result.shape[0]-8, 8):
        for j in range(0, result.shape[1]-8, 8):
            v_block[(i, j)] = []
            for m in range(0, 2):
                for n in range(0, 2):
                    for k in range(1, dis+1):
                        try:
                            v_block[(i, j)].append(v_cell[(i+m*8, j+n*8)][k])
                        except:
                            v_block[(i, j)].append(0)
            sum_num = 0
            for p in range(len(v_block[(i, j)])):
                sum_num += v_block[(i, j)][p]
            for p in range(len(v_block[(i, j)])):
                if sum_num == 0:
                    v_block[(i, j)][p] = 0
                else:
                    v_block[(i, j)][p] /= sum_num
    feature_space = []
    for key in v_block.keys():
        for i in range(len(v_block[key])):
            feature_space.append(round(v_block[key][i], 2))
    return feature_space, v_cell, result


'''
def draw_cell(v_cell, result):
    """
    ~画出特征图~
    分为9种情况,根据v_cell字典的值来绘出不同方向的图,由于方向梯度值最大的方向对轮廓影响最大,所以每个cell只画出一条线
    """
    # print(v_cell)
    view_cell = numpy.zeros((result.shape[1], result.shape[0]))
    def func(cell_dic): return sorted(
        cell_dic, key=cell_dic.__getitem__, reverse=True)[0]
    for key in v_cell.keys():
        max_key = func(v_cell[key])
        if max_key == 1:
            cv2.line(view_cell, (key[0]+4, key[1]+1),
                     (key[0]+4, key[1]+7), (255, 255, 255), 1)
        elif max_key == 2:
            cv2.line(view_cell, (key[0]+5, key[1]+1),
                     (key[0]+3, key[1]+7), (255, 255, 255), 1)
        elif max_key == 3:
            cv2.line(view_cell, (key[0]+6, key[1]+1),
                     (key[0]+2, key[1]+7), (255, 255, 255), 1)
        elif max_key == 4:
            cv2.line(view_cell, (key[0]+7, key[1]+2),
                     (key[0]+1, key[1]+6), (255, 255, 255), 1)
        elif max_key == 5:
            cv2.line(view_cell, (key[0]+7, key[1]+4),
                     (key[0]+1, key[1]+5), (255, 255, 255), 1)
        elif max_key == 6:
            cv2.line(view_cell, (key[0]+7, key[1]+5),
                     (key[0]+1, key[1]+4), (255, 255, 255), 1)
        elif max_key == 7:
            cv2.line(view_cell, (key[0]+7, key[1]+6),
                     (key[0]+1, key[1]+3), (255, 255, 255), 1)
        elif max_key == 8:
            cv2.line(view_cell, (key[0]+7, key[1]+7),
                     (key[0]+1, key[1]+2), (255, 255, 255), 1)
        elif max_key == 9:
            cv2.line(view_cell, (key[0]+7, key[1]+7),
                     (key[0]+1, key[1]+1), (255, 255, 255), 1)
    return view_cell
'''


def draw_cell(v_cell, result):
    '''
    ~画出特征图(改进后)~
    将原图方法20倍画出特征图,先设定最大长度为对角线长度,根据theta和规格化后的长度求出起点(x1, y1), 终点(x2, y2), 然后划线.
    规格化长度即根据原始值以max_length为单位求出一个新的值作为规格化后的长度.
    '''
    max_length = 160 * numpy.sqrt(2)
    view_cell = numpy.zeros((result.shape[0]*20, result.shape[1]*20))

    def get_sum(key, v_cell):
        sum_num = 0
        for inner_key in v_cell[key].keys():
            sum_num += v_cell[key][inner_key]
        return sum_num
    for key in v_cell.keys():
        center_point = (key[0]*20+80, key[1]*20+80)
        sum_num = get_sum(key, v_cell)
        if sum_num == 0:
            continue
        for inner_key in v_cell[key].keys():
            theta = ((inner_key-1)*20 + 10)*(numpy.pi/180)
            temp_length = numpy.ceil(
                (v_cell[key][inner_key] / sum_num) * max_length)
            if theta > (numpy.pi/2):
                print(max_length*numpy.cos(
                    theta))
                x1 = center_point[1] - abs(int(temp_length*numpy.cos(
                    theta)))
                y1 = center_point[0] - abs(int(temp_length*numpy.sin(theta)))
                x2 = center_point[1] + abs(int(temp_length*numpy.cos(
                    theta)))
                y2 = center_point[0] + abs(int(temp_length*numpy.sin(theta)))
            else:
                x1 = center_point[1] - int(temp_length*numpy.cos(
                    theta))
                y1 = center_point[0] + int(temp_length*numpy.sin(theta))
                x2 = center_point[1] + int(temp_length*numpy.cos(
                    theta))
                y2 = center_point[0] - int(temp_length*numpy.sin(theta))
            cv2.line(view_cell, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return view_cell


if __name__ == '__main__':
    feature_space, v_cell, result = get_hog(18)
    view_cell = draw_cell(v_cell, result).astype(numpy.uint8)
    cv2.imshow('pic', view_cell)
    # view_cell = draw_cell(v_cell, result).astype(numpy.uint8)
    # cv2.imshow('pic', view_cell)
    # img = cv2.imread('imageWork3/240be777a522347299bb9c221a077d42.jpg')
    # normalised_blocks, get_hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',visualize=True)
    # cv2.imshow('pic', get_hog_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
