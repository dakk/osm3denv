#version 330 core

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vTexCoord;

void main() {
    vWorldPos    = (p3d_ModelMatrix * p3d_Vertex).xyz;
    vWorldNormal = normalize((p3d_ModelMatrix * vec4(p3d_Normal, 0.0)).xyz);
    vTexCoord    = p3d_MultiTexCoord0;
    gl_Position  = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
