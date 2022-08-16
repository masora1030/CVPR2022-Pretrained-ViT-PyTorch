from OpenGL.GL import *
from OpenGL.GLU import *

def myGLDebugCallback(source, mtype, id, severity, length, message, userParam):
    print("[GLDBG]")
    if mtype == GL_DEBUG_TYPE_ERROR:
        raise SystemError("[GLDBG]")

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