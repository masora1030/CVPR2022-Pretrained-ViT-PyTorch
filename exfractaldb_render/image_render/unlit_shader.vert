#version 410

uniform mat4 mat_proj;
uniform mat4 mat_view;
uniform mat4 mat_model;

layout(location=0) in vec3 position;
layout(location=1) in vec4 color;

out vec4 fragInColor;

void main(){
    gl_Position = mat_proj * mat_view * mat_model * vec4(position, 1.0);
    fragInColor = color;
}