#version 410

uniform sampler2D colorSampler;

in vec2 fragInTexCoord;
out vec4 fragOutColor;

void main(){
    vec4 color = texture(colorSampler, fragInTexCoord);
    if(color.a < 0.1) discard;
    else fragOutColor = color;
    //else fragOutColor = vec4(1,0,0,1);
}
