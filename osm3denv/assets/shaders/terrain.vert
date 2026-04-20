#version 150

uniform mat4 world_view_proj_matrix;
uniform mat4 world_matrix;
uniform vec3 camera_position;
uniform float fog_start;
uniform float fog_end;

in vec4 position;
in vec3 normal;
in vec2 uv0;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out float v_fog_factor;

void main() {
    gl_Position = world_view_proj_matrix * position;
    vec4 wp = world_matrix * position;
    v_world_pos = wp.xyz;
    v_world_normal = mat3(world_matrix) * normal;
    v_uv = uv0;
    float d = length(camera_position - v_world_pos);
    v_fog_factor = clamp((fog_end - d) / max(fog_end - fog_start, 1e-3), 0.0, 1.0);
}
