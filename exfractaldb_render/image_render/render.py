import argparse
import pegl
import glm
import ctypes
import numpy as np
import math
import time
import os
import glob 
import random

from plyfile import PlyData, PlyElement
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from matplotlib import pyplot as plt
from util import createShader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_root", default="", type = str, help="load PLY root")
    parser.add_argument("--save_root", default="", type = str, help="save .png root")
    parser.add_argument("--zNear", default=1.0, type = float, help="")
    parser.add_argument("--zFar", default=100.0, type = float, help="")
    parser.add_argument("--view_point", default=12, type = int)
    parser.add_argument("--point_size", default=1, type = int)
    parser.add_argument("--img_size", default=224, type = int, help="image size")
    args = parser.parse_args()
    return args


def main(args):
    dpy = pegl.Display()
    pegl.bind_api(pegl.ClientAPI.OPENGL_API)
    conf = dpy.choose_config({
        pegl.ConfigAttrib.SURFACE_TYPE: pegl.SurfaceTypeFlag.PBUFFER_BIT,
        pegl.ConfigAttrib.BLUE_SIZE: 8,
        pegl.ConfigAttrib.GREEN_SIZE: 8,
        pegl.ConfigAttrib.RED_SIZE: 8,
        pegl.ConfigAttrib.DEPTH_SIZE: 16,
        })[0]
    ctx = conf.create_context()
    surf = conf.create_pbuffer_surface({pegl.SurfaceAttrib.WIDTH: args.img_size,
                                        pegl.SurfaceAttrib.HEIGHT: args.img_size})
    ctx.make_current(draw=surf)


    unlitShader_prog = createShader("./image_render/shader/unlit_shader.vert", "./image_render/shader/unlit_shader.frag")
    fbo = glGenFramebuffers(1)
    rbos = glGenRenderbuffers(2)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glBindRenderbuffer(GL_RENDERBUFFER, rbos[0])
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, args.img_size, args.img_size)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbos[0])
    glBindRenderbuffer(GL_RENDERBUFFER, rbos[1])
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, args.img_size, args.img_size)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbos[1])
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    pbo_color, pbo_depth = glGenBuffers(2)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
    glBufferData(GL_PIXEL_PACK_BUFFER, args.img_size*args.img_size*3, None, GL_STATIC_DRAW)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
    glBufferData(GL_PIXEL_PACK_BUFFER, args.img_size*args.img_size*4, None, GL_STATIC_DRAW)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

    glEnable(GL_DEPTH_TEST)
    axes_data = [
        0.0,0.0,0.0, 1.0,0.0,0.0,
        1.0,0.0,0.0, 1.0,0.0,0.0,
        0.0,0.0,0.0, 0.0,1.0,0.0,
        0.0,1.0,0.0, 0.0,1.0,0.0,
        0.0,0.0,0.0, 0.0,0.0,1.0,
        0.0,0.0,1.0, 0.0,0.0,1.0
    ]
    axes_vao = glGenVertexArrays(1)
    glBindVertexArray(axes_vao)
    axes_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, axes_vbo)
    glBufferData(GL_ARRAY_BUFFER, 4 * len(axes_data), (ctypes.c_float * len(axes_data))(*axes_data), GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(4*3))
    glBindVertexArray(0)

    cat_lists = sorted(os.listdir(args.load_root))
    for cat in cat_lists:
        cat_path = os.path.join(args.load_root, cat)
        ply_list = sorted(glob.glob(cat_path+"/*.ply"))

        for i, ply in enumerate(ply_list):
            with open(ply, 'rb') as f:
                plydata = PlyData.read(f)
                pts=np.array([
                np.asarray(plydata.elements[0].data['x']),
                np.asarray(plydata.elements[0].data['y']),
                np.asarray(plydata.elements[0].data['z'])
                ], dtype=np.float32).T

                pts_colors = np.ones((pts.shape[0], 3), dtype=np.float32)
                pts_vao = glGenVertexArrays(1)
                pts_vbo = glGenBuffers(2)
                glBindVertexArray(pts_vao)
                glBindBuffer(GL_ARRAY_BUFFER, pts_vbo[0])
                glBufferData(GL_ARRAY_BUFFER, 4 * pts.size, (ctypes.c_float * pts.size)(*pts.reshape(pts.size)), GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))
                glBindBuffer(GL_ARRAY_BUFFER, pts_vbo[1])
                glBufferData(GL_ARRAY_BUFFER, 4 * pts_colors.size, (ctypes.c_float * pts_colors.size)(*pts_colors.reshape(pts_colors.size)), GL_STATIC_DRAW)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4 * 3, ctypes.c_void_p(0))
                glBindVertexArray(0)

                board_data = [
                    0,-1, 0, 0,0,0,
                    0, 1, 0, 0,0,0,
                    1, 1, 0, 1,0,0,
                    1,-1, 0, 1,0,0,
                ]
                board_vao = glGenVertexArrays(1)
                board_vbo = glGenBuffers(1)
                glBindVertexArray(board_vao)
                glBindBuffer(GL_ARRAY_BUFFER, board_vbo)
                glBufferData(GL_ARRAY_BUFFER, 4 * len(board_data), (ctypes.c_float * len(board_data))(*board_data), GL_STATIC_DRAW)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(0))
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 4*6, ctypes.c_void_p(4*3))
                glBindVertexArray(0)

                mat_identity = glm.mat4(1)
                unlitShader_mat_proj_ul = glGetUniformLocation(unlitShader_prog, "mat_proj")
                unlitShader_mat_view_ul = glGetUniformLocation(unlitShader_prog, "mat_view")
                unlitShader_mat_model_ul = glGetUniformLocation(unlitShader_prog, "mat_model")
                glUseProgram(unlitShader_prog)
                glUniformMatrix4fv(unlitShader_mat_proj_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
                glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
                glUniformMatrix4fv(unlitShader_mat_model_ul, 1, GL_FALSE, glm.value_ptr(mat_identity))
                glUseProgram(0)

                mat_proj = glm.perspective(glm.radians(45.0), args.img_size / args.img_size, args.zNear, args.zFar)
                mat_proj_np = np.asarray(mat_proj, dtype=np.float32).reshape((4,4)).T
                mat_proj_np_inv = np.linalg.inv(mat_proj_np)

                gazePos = glm.vec3((0,0,0))
                camDist = 1
                camDir = glm.vec3((0,0,0))
                camPos  =gazePos - camDir * camDist
                upDir = glm.vec3((0, 1, 0))
                mat_view = glm.lookAt(camPos, gazePos, upDir)

                glUseProgram(unlitShader_prog)
                glUniformMatrix4fv(unlitShader_mat_proj_ul, 1, GL_FALSE, glm.value_ptr(mat_proj))
                glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_view))
                glUseProgram(0)

                fCount = 0
                while fCount <= (args.view_point - 1):
                    tbc = time.time()
                    gazePos = glm.vec3([0, 0, 0])
                    camDist = 4
                    theta= fCount / args.view_point*math.pi*2.0
                    mat_rot = glm.mat4(1)
                    mat_rot = glm.rotate(mat_rot, theta, glm.vec3(0,1,0))
                    mat_rot = glm.rotate(mat_rot, glm.radians(30), glm.vec3(0,1,0))
                    mat_rot = glm.rotate(mat_rot, glm.radians(random.uniform(-math.pi, math.pi)), glm.vec3(0,1,0))
                    camDir = (mat_rot * glm.vec4(1,0,0,0)).xyz
                    camPos = gazePos - camDir * camDist
                    upDir = (mat_rot * glm.vec4(0,1,0,0)).xyz
                    mat_view = glm.lookAt(camPos, gazePos, upDir)
                    
                    glUseProgram(unlitShader_prog)
                    glUniformMatrix4fv(unlitShader_mat_view_ul, 1, GL_FALSE, glm.value_ptr(mat_view))
                    glUseProgram(0)
                    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                    glClearColor(0.0, 0.0, 0.0, 0.0)
                    glViewport(0, 0, args.img_size, args.img_size)
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glUseProgram(unlitShader_prog)
                    glPointSize(args.point_size) 
                    glBindVertexArray(pts_vao)
                    glDrawArrays(GL_POINTS, 0, pts.shape[0])
                    glBindVertexArray(0)
                    glUseProgram(0)

                    glFlush()
                    glFinish()

                    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                    glReadPixels(0, 0, args.img_size, args.img_size, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
                    glReadPixels(0, 0, args.img_size, args.img_size, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

                    #RGB Image Rendering
                    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                    ret_color_ptr = ctypes.cast(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY), ctypes.POINTER(ctypes.c_ubyte))
                    ret_color = np.ctypeslib.as_array(ret_color_ptr, shape=(args.img_size, args.img_size, 3))
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                    img_color = Image.fromarray(ret_color)
                    os.makedirs(os.path.join(args.save_root, cat), exist_ok=True)
                    img_color.save(args.save_root + "/" + cat +"/"+ cat +"_{:05d}_{:03d}.png".format(i, fCount))

                    tcc = time.time()
                    print("duration: %.1f"%(tcc-tbc))
                    fCount=fCount+1

if __name__=='__main__':
    args = parse_args()
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    main(args)