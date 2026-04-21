#version 150

// Vertex shader for building meshes. Same structure as terrain.vert but also
// reads the per-vertex ``colour`` attribute so the fragment shader can tint
// each building's wall albedo to a distinct residential hue. The mesh
// generator sets a single palette colour across every vertex of one building.

uniform mat4 world_view_proj_matrix;
uniform mat4 world_matrix;

in vec4 position;
in vec3 normal;
in vec2 uv0;
in vec4 colour;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out vec4 v_color;

void main() {
    gl_Position = world_view_proj_matrix * position;
    vec4 wp = world_matrix * position;
    v_world_pos = wp.xyz;
    v_world_normal = mat3(world_matrix) * normal;
    v_uv = uv0;
    v_color = colour;
}
