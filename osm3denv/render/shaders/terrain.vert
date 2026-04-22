#version 330 core

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;

out vec3 vWorldPos;
out vec3 vWorldNormal;

void main() {
    vWorldPos    = (p3d_ModelMatrix * p3d_Vertex).xyz;
    vWorldNormal = normalize((p3d_ModelMatrix * vec4(p3d_Normal, 0.0)).xyz);
    gl_Position  = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
