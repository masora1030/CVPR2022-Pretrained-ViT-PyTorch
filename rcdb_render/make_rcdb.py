import math
from PIL import Image, ImageDraw
import random
import noise
import os
import argparse
import csv

def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default="./dataset",type=str, help="path to image file save directory")
    parser.add_argument("--numof_classes", default=1000, type=int, help="RCDB category number")
    parser.add_argument("--numof_instances", default=1000, type=int, help="RCDB instance number")
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--numof_thread", default=0, type=int, help="")
    parser.add_argument("--thread_num", default=0, type=int, help="")
    # Category parameter setting
    parser.add_argument("--vertex_num", default=200, type=int, help="")
    parser.add_argument("--perlin_min", default=0, type=int, help="")
    parser.add_argument("--line_width", default=0.1, type=float, help="")
    parser.add_argument("--radius_min", default=0, type=int, help="")
    parser.add_argument("--line_num_min", default=1, type=int, help="")
    parser.add_argument("--oval_rate", default=2, type=int, help="")
    parser.add_argument("--start_pos", default=400, type=int, help="")

    # Display on screen
    parser.add_argument("--display", action='store_true', help="Display the generated images")
    args = parser.parse_args()
    return args

args = conf()


vertex_x = []
vertex_y = []
Noise_x = []
Noise_y = []
im = []
vertex_number = 2
random.seed(args.thread_num + 1)

class_per_thread = args.numof_classes / args.numof_thread
cat_start = args.thread_num * int(class_per_thread)
cat_finish = cat_start + int(class_per_thread)

for cat in range(int(cat_start), int(cat_finish)):

    if not os.path.exists(os.path.join(args.save_root, "image/%05d" % cat)):
        os.makedirs(os.path.join(args.save_root, "image/%05d" % cat))
    if not os.path.exists(os.path.join(args.save_root, "param")):
        os.makedirs(os.path.join(args.save_root, "param" ))

    # Prameter search per category
    while True:
        vertex_number = int(random.expovariate(1 / 100))

        if (vertex_number > 2 and vertex_number <= args.vertex_num):
            break

    line_draw_num = random.randint(args.line_num_min , args.line_num_min + 199)
    perlin_noise_coefficient = random.uniform(args.perlin_min, (args.perlin_min + 4))
    line_width = random.uniform(0.0, args.line_width)
    start_rad = random.randint(args.radius_min, args.radius_min + 100)
    oval_rate_x = random.uniform(1, args.oval_rate)
    oval_rate_y = random.uniform(1, args.oval_rate)
    start_pos_h = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2
    start_pos_w = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2

    # csv file save
    with open(os.path.join(args.save_root, "param/%05d.csv" % cat), 'w') as f:
        param = {'Vertex': vertex_number, 'Perlin_noise': perlin_noise_coefficient, 'line_width': line_width, \
                    'Center_rad': start_rad, 'Line_num': line_draw_num, 'Oval_rate_x': oval_rate_x, 'Oval_rate_y': oval_rate_y, \
                    'start_pos_h': start_pos_h, 'start_pos_w': start_pos_w}
        writer = csv.writer(f)
        for k, v in param.items():
            writer.writerow([k, v])

    vertex_x.clear()
    vertex_y.clear()
    im.clear()

    for k2 in range(args.numof_instances):

        im.append(Image.new('RGB', (args.image_size, args.image_size), (255, 255, 255)))
        draw = ImageDraw.Draw(im[k2])
        angle = (math.pi * 2) / vertex_number
    
        for vertex in range(vertex_number):
            vertex_x.append(math.cos(angle * vertex)* start_rad * oval_rate_x)
            vertex_y.append(math.sin(angle * vertex)* start_rad * oval_rate_y)
    
        vertex_x.append(vertex_x[0])
        vertex_y.append(vertex_y[0])

        for line_draw in range(line_draw_num):
            r = random.randint(0, args.image_size / 2)

            Noise_x.clear()
            Noise_y.clear()
            for vertex in range(vertex_number):
                Noise_x.append(random.uniform(0 , 10000))
                Noise_x[vertex] = noise.pnoise1(Noise_x[vertex]) * perlin_noise_coefficient * 2 - perlin_noise_coefficient

            for vertex in range(vertex_number):
                Noise_y.append(random.uniform(0 , 10000))
                Noise_y[vertex] = noise.pnoise1(Noise_y[vertex]) * perlin_noise_coefficient * 2 - perlin_noise_coefficient

            for vertex in range(vertex_number):
                vertex_x[vertex] -= math.cos(angle * vertex) * (Noise_x[vertex] - line_width)
                vertex_y[vertex] -= math.sin(angle * vertex) * (Noise_y[vertex] - line_width)

            vertex_x[vertex_number] = vertex_x[0]
            vertex_y[vertex_number] = vertex_y[0]

            for i in range(vertex_number):
                draw.line((vertex_x[i] + start_pos_w, vertex_y[i] + start_pos_h, vertex_x[i + 1] + start_pos_w, vertex_y[i + 1] + start_pos_h), fill = (r,r,r), width = 1)

        if not args.display:
            im[k2].save(args.save_root + "/image/%05d/%05d_%04d.png" % (cat, cat, k2), quality = 95)
        else:
            im[k2].show()
        
        vertex_x.clear()
        vertex_y.clear()

        oval_rate_x = random.uniform(1, args.oval_rate)
        oval_rate_y = random.uniform(1, args.oval_rate)
        start_rad = random.randint(args.radius_min, args.radius_min + 100)
        start_pos_h = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2
        start_pos_w = (args.image_size + random.randint(-1 * args.start_pos, args.start_pos)) / 2

    print('Gerated Category:' + str(cat))

