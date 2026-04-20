#version 150

uniform mat4 world_view_proj_matrix;
uniform mat4 world_matrix;
uniform vec3 camera_position;
uniform float time;
uniform float fog_start;
uniform float fog_end;

in vec4 position;
in vec2 uv0;

out vec3 v_world_pos;
out vec2 v_uv_a;
out vec2 v_uv_b;
out float v_fog_factor;

void main() {
    gl_Position = world_view_proj_matrix * position;
    vec4 wp = world_matrix * position;
    v_world_pos = wp.xyz;
    v_uv_a = uv0 * 0.05 + vec2(time * 0.010, time * 0.006);
    v_uv_b = uv0 * 0.08 + vec2(-time * 0.007, time * 0.012);
    float d = length(camera_position - v_world_pos);
    v_fog_factor = clamp((fog_end - d) / max(fog_end - fog_start, 1e-3), 0.0, 1.0);
}
