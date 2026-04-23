#version 330 core

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// Remap UV [0,1]×[0,1] → card position [-1,1]×[-1,1] (Y+ = up = forward direction)
out vec2 vCardPos;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vCardPos    = p3d_MultiTexCoord0 * 2.0 - 1.0;
}
