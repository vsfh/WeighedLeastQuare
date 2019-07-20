import os
from skimage.feature import hog
from PIL import Image

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
    
from numpy import *
import numpy as np


class POISSON(object):
    """the POISSON module"""

    def __str__(self):
        return "\nUsing the algorithm POISSON.....\n"

    def get_name(self):
        return "POISSON"    

    #read the configure file    
    def get_options(self):
        cf = ConfigParser.ConfigParser()
        cf.read("config.cof")

        option_dict = dict()

        for key, value in cf.items("POISSON"):

            option_dict[key] = eval(value)

        # print(option_dict)
        return option_dict
    
    #read image

    def lwlr(self):
        option = self.get_options()
        xArr = option["xarr"]; yArr = option["yarr"]
        a_1 = option["a_1"]; b_1 = option["b_1"]

        xMat = mat(xArr); yMat = mat(yArr).T
        m = shape(xMat)[0]#表示x中样本的个数
        weights = mat(eye((m)))#初始化为单位矩阵
        for j in range(m):                      #下面两行用于创建权重矩阵
            diffMat = abs(yArr[j] - a_1 + b_1 * xArr[j][1])/(1 + b_1 * b_1)**0.5#样本点与待预测点的距离
            weights[j,j] = math.exp(-1 * diffMat)
        xTx = xMat.T * (weights * xMat)
        if linalg.det(xTx) == 0.0:#判断行列式是否为满秩
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return ws

    def get_a(self):
        return self.lwlr()[1,0]
        
    def get_b(self):
        return self.lwlr()[0,0]



if __name__ == '__main__':
    lwlr()

