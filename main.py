from PIL import Image
import os
import time
import numpy as np
from functools import partial
import multiprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sub_modules import my_Poisson
from sub_modules import my_Normal

from matplotlib import pyplot as plt
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")


try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

try:
    import cPickle as pickle
except ImportError:
    import pickle



def create_visualization_dir():
    """create the visualization directory
    """
    if not os.path.exists("visualization"):
        os.mkdir("visualization")
    else:
        import shutil
        shutil.rmtree("visualization")
        os.mkdir("visualization")


class ImageProcess(object):
    """
    the main class of pyHIVES    
    """

    def __init__(self):
        self.option_dict = self.get_options()
        # print(self.option_dict)

    def get_options(self):
        """read the configure file
           this function generate a dict，all the parameters are the items of dict 
        """
        cf = ConfigParser.ConfigParser()

        if os.path.exists("config.cof"):
            cf.read('config.cof')
        else:
            print("there is no config.cof!")
            exit()

        option_dict = dict()
        for key, value in cf.items("MAIN"):
            option_dict[key] = eval(value)

        return option_dict

    def get_algorithm(self):
        """algorithm selecting function
           there are five different algorithms,
           this function select the algorithm 
           through the configure file.
        """
        algorithms = []
        for algorithm in self.option_dict["algorithm"]:
            if algorithm == "Poisson":
                algorithms.append(my_Poisson.POISSON())
            if algorithm == "Normal":
                algorithms.append(my_Normal.NORMAL())

        return algorithms

    def image_generate(self,a,b,file,algorithm):
        """draw the histogram
        """
        x = np.arange(0.0 ,40, 10)
        plt.figure(1,figsize = (8,6))
        plt.plot(x, a * x + b)
        plt.savefig(os.path.join("visualization",file.split(".")[0] + "_" + algorithm.get_name()))
        plt.close()

    def run(self):
        """ high-level function to run the entire class.
        """ 
        create_visualization_dir()
        algorithm_list = self.get_algorithm()

        # extracting features with single process
        for algorithm in algorithm_list:
            a = algorithm.get_a()
            b = algorithm.get_b()
            self.image_generate(a,b,"images",algorithm) 



if __name__ == '__main__':
    start = time.time()
    processor = ImageProcess()
    processor.run()
    end = time.time()
    print(start - end)
