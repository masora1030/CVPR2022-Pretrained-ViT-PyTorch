# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
from IteratedFunctionSystem import ifs_function
import open3d

def conf():
	parser = argparse.ArgumentParser()
	parser.add_argument("--load_root", default="./CVPR2022_fractal_param_search/var_0.1_cat10k/fractal_obj_param_list", type = str, help="load csv root")
	parser.add_argument("--save_root", default="./PC-Fractal-object/var0.10/test_sample", type = str, help="save PLY root")
	parser.add_argument("--draw_type", default="point_gray", type = str, help="point_gray, point_color, patch_gray")
	parser.add_argument("--point_num", default=10000, type = int)
	parser.add_argument("--classes", default=1000, type = int)
	parser.add_argument("--normalize", default=1.0, type=float)
	args = parser.parse_args()
	return args

def min_max(args, x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = ((x-min)/(max-min)) * (args.normalize - (-args.normalize)) - args.normalize
    return result

def centoroid(point):
    new_centor = []
    sum_x = (sum(point[0]) / args.point_num)
    sum_y = (sum(point[1]) / args.point_num)
    sum_z = (sum(point[2]) / args.point_num)
    centor_of_gravity = [sum_x, sum_y, sum_z]
    # centor_of_gravity = centor_of_gravity.reshape([3,1])
    fractal_point_x = (point[0] - centor_of_gravity[0]).tolist()
    fractal_point_y = (point[1] - centor_of_gravity[1]).tolist()
    fractal_point_z = (point[2] - centor_of_gravity[2]).tolist()
    new_centor.append(fractal_point_x)
    new_centor.append(fractal_point_y)
    new_centor.append(fractal_point_z)
    new = np.array(new_centor)
    return new

def generator(args, params):
    generators = ifs_function()
    for param in params:
        generators.set_param(float(param[0]), float(param[1]),
                             float(param[2]), float(param[3]),
                             float(param[4]), float(param[5]),
                             float(param[6]), float(param[7]),
                             float(param[8]), float(param[9]),
                             float(param[10]), float(param[11]),
                             float(param[12]),weight_a=float(weight[0]),
                             weight_b=float(weight[1]),weight_c=float(weight[2]),
                             weight_d=float(weight[3]),weight_e=float(weight[4]),
                             weight_f=float(weight[5]),weight_g=float(weight[6]),
                             weight_h=float(weight[7]),weight_i=float(weight[8]),
                             weight_j=float(weight[9]),weight_k=float(weight[10]),
                             weight_l=float(weight[11]))
    data = generators.calculate(args.point_num + 1)
    return data

if __name__ == "__main__":
	starttime = time.time()
	args = conf()

	if not os.path.exists(args.save_root):
		os.makedirs(args.save_root)

	csv_names = os.listdir(args.load_root)
	csv_names.sort()
	weights = np.genfromtxt("./3dfractal_render/ifs_weight/weights_ins145.csv",dtype=np.str,delimiter=",")
	for i, csv_name in enumerate(csv_names):
		name, ext = os.path.splitext(csv_name)
		
		if i == args.classes:
			break

		if ext != ".csv":
			continue
		print(name)

		root = os.path.join(args.save_root, name)
		if not os.path.exists(root):
			os.mkdir(root)
		fractal_weight = 0
		for weight in weights:
			padded_fractal_weight= '%04d' % fractal_weight
			params = np.genfromtxt(args.load_root+"/"+csv_name,dtype=np.str,delimiter=",")
			if args.draw_type == "point_gray":
				generators = ifs_function()
				for param in params:
					generators.set_param(float(param[0]), float(param[1]),float(param[2]), float(param[3]),
						float(param[4]), float(param[5]),float(param[6]), float(param[7]),
						float(param[8]), float(param[9]),float(param[10]), float(param[11]),
						float(param[12]),weight_a=float(weight[0]),weight_b=float(weight[1]),weight_c=float(weight[2]),
						weight_d=float(weight[3]),weight_e=float(weight[4]),weight_f=float(weight[5]),weight_g=float(weight[6]),
						weight_h=float(weight[7]),weight_i=float(weight[8]),weight_j=float(weight[9]),weight_k=float(weight[10]),
						weight_l=float(weight[11]))
				fractal_point = generators.calculate(args.point_num)
				point = min_max(args, fractal_point, axis=None)
				point = centoroid(point)
				arr = np.isnan(point).any(axis=1)
				if arr[1] == False:
					point_data = point.transpose()
					pointcloud = open3d.geometry.PointCloud()
					pointcloud.points = open3d.utility.Vector3dVector(point_data)
					open3d.io.write_point_cloud((root +"/"+name+"_"+padded_fractal_weight+".ply"), pointcloud)
			fractal_weight += 1

	endtime = time.time()
	interval = endtime - starttime
	print("passed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
