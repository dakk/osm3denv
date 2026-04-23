#version 330 core

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

uniform mat4 p3d_ModelViewProjectionMatrix;

out vec2 v_uv;   // (azimuth/2π, elevation/halfπ)
out vec3 v_dir;  // normalized direction — used as seamless 3-D noise domain

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_uv  = p3d_MultiTexCoord0;
    v_dir = normalize(p3d_Vertex.xyz);
}
