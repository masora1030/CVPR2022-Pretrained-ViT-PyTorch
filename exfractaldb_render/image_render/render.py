# depends: PySDL2, PyOpenGL, plyfile, OpenEXR, PIL
# Moreover, Dynamic Link Libraries of Original SDL2 required.
# Setting paths also reuired:
# export PYSDL2_DLL_PATH="/path/to/SDL2/bin"
# export PATH="/path/to/SDL2/bin:$PATH"

from sdl2 import *
from OpenGL.GL import *
from OpenGL.GLU import *
import glm
import ctypes
from plyfile import PlyData, PlyElement
import numpy as np
import OpenEXR, array, Imath
from PIL import Image
import math
import time
import sdl2.ext
from matplotlib import pyplot as plt
import os
import cv2
import glob 
import random
from PIL import Image

def conf():
	parser = argparse.ArgumentParser()
	parser.add_argument("--load_root", default='./3dfractal_render/ifs_weight/weights_ins145.csv', type = str, help="load PLY root")
	parser.add_argument("--save_root", default="./dataset/EXFractalDB", type = str, help="save .png root")
	args = parser.parse_args()
	return args

def myGLDebugCallback(source, mtype, id, severity, length, message, userParam):
    print("[GLDBG]")
    if mtype == GL_DEBUG_TYPE_ERROR:
        raise SystemError("[GLDBG]")

class TextDrawer:
    tex = None
    sfc = None
    Width = 512
    Height = 512
    font = None
    printTexShader_prog = None
    vao = None
    indices_num = 0
    colorSampler_ul = None
    def __init__(self, renderer):
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.sfc = SDL_CreateRGBSurface(0, self.Width, self.Height, 32, 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff)
        self.font = sdl2.ext.FontManager("./image_render/arial.ttf", size=16, color=sdl2.ext.Color(255, 255, 255, 255), bg_color=sdl2.ext.Color(0, 0, 0, 0))

        points=np.array([
            -1,-1,-1,
            1,-1,-1,
            1,1,-1,
            -1,1,-1
        ],dtype=np.float32)
        UVs=np.array([
            0,1,
            1,1,
            1,0,
            0,0
        ],dtype=np.float32)
        indices=np.array([
            0,1,2,
            2,3,0
        ],dtype=np.uint32)
        self.indices_num = indices.size
        self.vao = glGenVertexArrays(1)
        points_vbo, UVs_vbo, indices_vbo = glGenBuffers(3)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, points_vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*len(points), points, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0 ,3, GL_FLOAT, GL_FALSE, 4*3, ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, UVs_vbo)
        glBufferData(GL_ARRAY_BUFFER, 4*len(UVs), UVs, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1 ,2, GL_FLOAT, GL_FALSE, 4*2, ctypes.c_void_p(0))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indices_vbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(indices), indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

        self.printTexShader_prog = createShader("./image_render/printTexShader.vert","./image_render/printTexShader.frag")
        self.colorSampler_ul = glGetUniformLocation(self.printTexShader_prog, "colorSampler")
        glUseProgram(self.printTexShader_prog)
        glUniform1i(self.colorSampler_ul, 0)
        glUseProgram(0)
    def draw(self, text, position, size=None, color=None):
        SDL_FillRects(self.sfc,SDL_Rect(0,0,self.Width,self.Height),1,sdl2.ext.Color(0,0,0,0))
        text_surface = self.font.render(text, size=size, color=color)
        rect = SDL_Rect()
        SDL_GetClipRect(text_surface, rect)
        SDL_BlitScaled(text_surface, None, self.sfc, SDL_Rect(rect.x+position[0], rect.y+position[1], rect.w, rect.h))
        SDL_FreeSurface(text_surface)
        glUseProgram(self.printTexShader_prog)
        glActiveTexture(GL_TEXTURE0)
        glDisable(GL_DEPTH_TEST)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.Width, self.Height, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.cast(self.sfc.contents.pixels,ctypes.POINTER(ctypes.c_ubyte)))
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.indices_num, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_DEPTH_TEST)
        glUseProgram(0)

def createShader(vertFile, fragFile):
    with open(vertFile,'r',encoding='utf8') as fp:
        vertShader_code = fp.read()
    vertShader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertShader, [vertShader_code])
    glCompileShader(vertShader)
    compiledStatus = glGetShaderiv(vertShader, GL_COMPILE_STATUS)
    infoLog = glGetShaderInfoLog(vertShader)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if compiledStatus == GL_FALSE:
        raise Exception("Compile error in vertex shader.")

    with open(fragFile,'r',encoding='utf8') as fp:
        fragShader_code = fp.read()
    fragShader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragShader, [fragShader_code])
    glCompileShader(fragShader)
    compiledStatus = glGetShaderiv(fragShader, GL_COMPILE_STATUS)
    infoLog = glGetShaderInfoLog(fragShader)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if compiledStatus == GL_FALSE:
        raise Exception("Compile error in fragment shader.")

    shader_prog = glCreateProgram()
    glAttachShader(shader_prog, vertShader)
    glAttachShader(shader_prog, fragShader)
    glDeleteShader(vertShader)
    glDeleteShader(fragShader)

    glLinkProgram(shader_prog)
    shader_linked = ctypes.c_uint(0)
    glGetProgramiv(shader_prog, GL_LINK_STATUS, ctypes.pointer(shader_linked))
    infoLog = glGetProgramInfoLog(shader_prog)
    if infoLog != '':
        print(infoLog.decode('ascii'))
    if shader_linked == GL_FALSE:
        raise Exception("Link error.")

    return shader_prog


Width = 512
Height = 512
zNear=0.1
zFar=100.0


SDL_Init(SDL_INIT_VIDEO)
SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4)
SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1)
SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG)

# window = SDL_CreateWindow(b"render_opengl",
#                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
#                               Width, Height, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL)
window = SDL_CreateWindow(b"Hello World",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              Width, Height, SDL_WINDOW_HIDDEN | SDL_WINDOW_OPENGL)

context = SDL_GL_CreateContext(window)

print("GL VERSION: " + glGetString(GL_VERSION).decode('utf8'))
print('glDebugMessageCallback Available: %s' % bool(glDebugMessageCallback))
gl_major_version = glGetInteger(GL_MAJOR_VERSION)
gl_minor_version = glGetInteger(GL_MAJOR_VERSION)
gl_version= gl_major_version+gl_minor_version/10

renderer = SDL_CreateRenderer(window,-1, SDL_RENDERER_ACCELERATED)
textDrawer = TextDrawer(renderer)

unlitShader_prog = createShader("./image_render/unlit_shader.vert","./image_render/unlit_shader.frag")

fbo = glGenFramebuffers(1)
rbos = glGenRenderbuffers(2)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glBindRenderbuffer(GL_RENDERBUFFER, rbos[0])
glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, Width, Height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbos[0])
glBindRenderbuffer(GL_RENDERBUFFER, rbos[1])
glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, Width, Height)
glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbos[1])
glBindFramebuffer(GL_FRAMEBUFFER, 0)

pbo_color, pbo_depth = glGenBuffers(2)
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
glBufferData(GL_PIXEL_PACK_BUFFER, Width*Height*3, None, GL_STREAM_COPY)
glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
glBufferData(GL_PIXEL_PACK_BUFFER, 4*Width*Height, None, GL_STREAM_COPY)
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

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

cat_list = sorted(os.listdir(args.load_root))
for cat_ in cat_list:
    cat_path = os.path.join(args.load_root, cat_)
    ply_list = sorted(glob.glob(cat_path+"/*.ply"))

    if not os.path.exists(args.save_path + "/" + cat_):
        os.makedirs(args.save_path + "/" + cat_)
    # if not os.path.exists(save_path + "/depth/" + cat_):
    #     os.makedirs(save_path + "/depth/" + cat_)

    for i, ply in enumerate(ply_list):
        with open(ply, 'rb') as f:
            plydata = PlyData.read(f)
            pts=np.array([
            np.asarray(plydata.elements[0].data['x']),
            np.asarray(plydata.elements[0].data['y']),
            np.asarray(plydata.elements[0].data['z'])
            ], dtype=np.float32).T

            pts_colors = np.ones((pts.shape[0], 3), dtype=np.float32)
            #
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

            mat_proj = glm.perspective(glm.radians(45.0), Width / Height, zNear, zFar)
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


            def backprojf(z):
                c=mat_proj_np_inv[2,2]
                d=mat_proj_np_inv[2,3]
                e=mat_proj_np_inv[3,2]
                f=mat_proj_np_inv[3,3]
                return - (c*z+d) / (e*z+f)
            backproj = np.frompyfunc(backprojf,1,1)

            fCount = 0
            bfCount = 0
            loopFlg = True
            fps_text="fps: "
            bc = time.time()
            while fCount <= 11:
                print(fCount)
                tbc = time.time()
                gazePos = glm.vec3([0, 0, 0])
                camDist = 4
                theta= fCount/12*math.pi*2.0
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
                glViewport(0, 0, Width, Height)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                glUseProgram(unlitShader_prog)
                glPointSize(3) 
                glBindVertexArray(pts_vao)
                glDrawArrays(GL_POINTS, 0, pts.shape[0])
                glBindVertexArray(0)
                glUseProgram(0)

                glFlush()

                # glBindFramebuffer(GL_FRAMEBUFFER, 0)
                # glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
                # glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
                # #glReadBuffer(GL_FRONT)
                # glBlitFramebuffer(
                #     0,0,Width,Height,
                #     0,0,Width,Height,
                #     GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT,
                #     GL_NEAREST
                # )
                # glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

                SDL_GL_SwapWindow(window)

                glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                glReadPixels(0, 0, Width, Height, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_depth)
                glReadPixels(0, 0, Width, Height, GL_DEPTH_COMPONENT, GL_FLOAT, ctypes.c_void_p(0))
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

                #color
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_color)
                ret_color_ptr = ctypes.cast(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY), ctypes.POINTER(ctypes.c_ubyte))
                ret_color = np.ctypeslib.as_array(ret_color_ptr, shape=(Height,Width,3))
                print(ret_color.shape)
                print(ret_color.dtype)
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                img_color = Image.fromarray(ret_color)
                img_color.save(args.save_path + "/" + cat_ +"/"+ cat_ +"_{:05d}_{:03d}.png".format(i, fCount))

                tcc = time.time()
                print("duration: %.1f"%(tcc-tbc))

                fCount=fCount+1