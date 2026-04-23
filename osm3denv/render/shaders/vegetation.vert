#version 330 core
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
out vec4 vColor;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vColor = p3d_Color;
}
