import numpy as np
from error_propagation import *

from fitting_and_analysis import CurveFitFuncs
cff = CurveFitFuncs()

class DataLoader():
    def __init__(self, filename):

        floor_to_r_factor = 3.95
        mgal_to_g_factor = 1 / 980665

        min_range = 0
        max_range = -1

        directory = '%s.txt' % (filename)
        self.full_data = np.loadtxt(directory, delimiter=',', dtype=float)[min_range:max_range].T

        floors = self.full_data[0]
        left_edge = self.full_data[1]
        middle = self.full_data[2]
        right_edge = self.full_data[3]
        
        errors = np.abs(right_edge - left_edge) /4

        delta_r_arr = []
        delta_g_arr = []
        delta_g_error_arr = []
        
        for i in range(len(floors)):
            for j in range(i, len(floors)):
                delta_r = floors[j] - floors[i]
                delta_g = middle[j] - middle[i]
                delta_g_error = np.sqrt(errors[j]**2 + errors[i]**2)
                
                delta_r_arr.append(delta_r)
                delta_g_arr.append(delta_g)
                delta_g_error_arr.append(delta_g_error)
                
        delta_r_arr = np.array(delta_r_arr)
        delta_g_arr = np.array(delta_g_arr)
        delta_g_error_arr = np.array(delta_g_error_arr)

        delta_rs = np.array(np.sort(list(set(delta_r_arr))))
        delta_r_errors = np.zeros_like(delta_rs) + 0

        delta_g_averages = []
        delta_g_average_errors = []

        for dr in delta_rs:
            delta_g_averages.append(np.average(delta_g_arr[delta_r_arr == dr]))
            delta_g_average_errors.append(average_error(delta_g_error_arr[delta_r_arr == dr]))

        delta_g_averages = np.array(delta_g_averages)
        delta_g_average_errors = np.array(delta_g_average_errors)

        self.y = delta_g_averages * mgal_to_g_factor
        self.y_error = delta_g_average_errors * mgal_to_g_factor
        
        self.x = delta_rs * floor_to_r_factor
        self.x_error = delta_r_errors * floor_to_r_factor

