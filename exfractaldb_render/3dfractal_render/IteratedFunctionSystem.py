import open3d
import numpy as np
import os
import random

class ifs_function():
    def __init__(self):
        self.prev_x,self.prev_y, self.prev_z = 0.0, 0.0, 0.0
        self.function  = []
        self.xs,self.ys, self.zs = [],[],[]
        self.select_function = []
        self.temp_proba = 0.0

    def set_param(self,a,b,c,d,e,f,g,h,i,j,k,l,proba, **kwargs):
        if "weight_a" in kwargs:
            a *= kwargs["weight_a"]
        if "weight_b" in kwargs:
            b *= kwargs["weight_b"]
        if "weight_c" in kwargs:
            c *= kwargs["weight_c"]
        if "weight_d" in kwargs:
            d *= kwargs["weight_d"]
        if "weight_e" in kwargs:
            e *= kwargs["weight_e"]
        if "weight_f" in kwargs:
            f *= kwargs["weight_f"]
        if "weight_g" in kwargs:
            g *= kwargs["weight_g"]
        if "weight_h" in kwargs:
            h *= kwargs["weight_h"]
        if "weight_i" in kwargs:
            i *= kwargs["weight_i"]
        if "weight_j" in kwargs:
            j *= kwargs["weight_j"]
        if "weight_k" in kwargs:
            k *= kwargs["weight_k"]
        if "weight_l" in kwargs:
            l *= kwargs["weight_l"]
        temp_function  = {"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"g":g,"h":h,"i":i,"j":j,"k":k,"l":l,"proba":proba}
        self.function.append(temp_function)
        self.temp_proba += proba
        self.select_function.append(self.temp_proba)


    def calculate(self,iteration):
        """ Recursively calculate coordinates for args.iteration """
        rand = np.random.random(iteration)
        select_function = self.select_function
        function = self.function
        prev_x, prev_y, prev_z = self.prev_x, self.prev_y, self.prev_z
        for i in range(iteration-1):
            for j in range(len(select_function)):
                if rand[i] <= select_function[j]:
                    next_x = prev_x*function[j]["a"] + \
                            prev_y*function[j]["b"] + \
                            prev_z*function[j]["c"] + \
                            function[j]["j"]
                    next_y = prev_x*function[j]["d"] + \
                            prev_y*function[j]["e"] + \
                            prev_z*function[j]["f"] + \
                            function[j]["k"]
                    next_z = prev_x*function[j]["g"] + \
                            prev_y*function[j]["h"] + \
                            prev_z*function[j]["i"] + \
                            function[j]["l"]
                    break
            self.xs.append(next_x), self.ys.append(next_y), self.zs.append(next_z)
            prev_x, prev_y, prev_z = next_x, next_y, next_z
        point_data = np.array((self.xs,self.ys,self.zs), dtype = float)
        return point_data
