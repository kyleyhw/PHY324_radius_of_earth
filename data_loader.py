import numpy as np


from fitting_and_analysis import CurveFitFuncs
cff = CurveFitFuncs()

class DataLoader():
    def __init__(self, filename):
        directory = '%s.txt' % (filename)
        self.full_data = np.loadtxt(directory, delimiter=',', dtype=float).T

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
                
        self.y = delta_g_arr
        self.y_error = delta_g_error_arr
        
        self.x = delta_r_arr
        self.x_error = np.zeros_like(delta_r_arr) 

