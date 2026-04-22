#version 150

// Fullscreen quad vertex shader used by every post-processing pass. The quad
// geometry is supplied by Ogre's render_quad compositor pass: positions come
// in already at clip-space [-1,1] and UVs at [0,1].

in vec4 position;
in vec2 uv0;

out vec2 v_uv;

void main() {
    gl_Position = position;
    v_uv = uv0;
}
