#version 150

// Vertex shader for procedural trees. Wind is applied only to vertices with
// a high ``sway`` factor (crown). uv0.x = sway [0, 1] — set by the mesh
// generator; uv0.y = per-tree seed in [0, 1) for per-tree wind phase.

uniform mat4 world_view_proj_matrix;
uniform mat4 world_matrix;
uniform float time;

in vec4 position;
in vec3 normal;
in vec2 uv0;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 wp = world_matrix * position;

    // Wind: per-tree phase offset + two-octave sine sway.
    float phase = time * 1.4 + uv0.y * 62.8;
    float sway = smoothstep(0.0, 1.0, uv0.x);
    vec2 wind = vec2(sin(phase) + 0.3 * sin(phase * 2.3 + 1.1),
                     cos(phase * 0.9) + 0.3 * sin(phase * 1.7));
    // Crown amplitude ~25 cm at max sway; trunk barely moves.
    wp.xz += wind * sway * 0.25;

    gl_Position = world_view_proj_matrix * vec4(wp.xyz, 1.0);
    v_world_pos = wp.xyz;
    v_world_normal = mat3(world_matrix) * normal;
    v_uv = uv0;
}
