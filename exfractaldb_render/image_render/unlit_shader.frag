#version 410

in vec4 fragInColor;
out vec4 fragOutColor;

void main(){
    if(fragInColor.a == 0) discard;
    else fragOutColor = fragInColor;
}