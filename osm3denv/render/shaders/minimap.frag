#version 330 core

uniform sampler2D u_map_tex;
uniform vec2  u_cam_uv;    // camera UV in stitched texture (V=0 south, V=1 north)
uniform float u_heading;   // radians, 0=north, positive=CCW (west)
uniform float u_map_scale; // fraction of stitched texture to display (1/n_tiles)

in vec2 vCardPos;   // [-1,1]×[-1,1]; Y+ = up = forward on heading-up map
out vec4 fragColor;

void main() {
    float d = length(vCardPos);
    if (d > 1.0) discard;

    // Rotate card position CCW by heading so the forward direction is always up.
    float c = cos(u_heading);
    float s = sin(u_heading);
    vec2 rot = vec2(vCardPos.x * c - vCardPos.y * s,
                    vCardPos.x * s + vCardPos.y * c);

    // rot in [-1,1] maps to half a tile in each direction around the camera.
    vec2 uv = u_cam_uv + rot * (u_map_scale * 0.5);
    vec3 col = texture(u_map_tex, uv).rgb;

    // Soft vignette near edge
    col *= 1.0 - smoothstep(0.72, 1.0, d) * 0.28;

    // Dark inner border ring, then lighter rim
    float f1 = smoothstep(0.90, 0.94, d);
    col = mix(col, vec3(0.06, 0.06, 0.08), f1);
    float f2 = smoothstep(0.94, 1.0, d);
    col = mix(col, vec3(0.65, 0.65, 0.68), f2);

    fragColor = vec4(col, 1.0);
}
