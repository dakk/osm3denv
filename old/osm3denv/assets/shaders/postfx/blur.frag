#version 150

// Separable 9-tap gaussian. ``direction`` is set by the material to (1, 0)
// for the horizontal pass and (0, 1) for the vertical pass.

uniform sampler2D src;
uniform vec2 direction;
uniform vec2 texel_size;   // 1 / target_resolution, set by material

in vec2 v_uv;
out vec4 frag_color;

void main() {
    // 9-tap gaussian weights (sigma ~ 2.5 px) normalised to sum 1.
    const float w0 = 0.227027;
    const float w1 = 0.194594;
    const float w2 = 0.121622;
    const float w3 = 0.054054;
    const float w4 = 0.016216;

    vec2 step = direction * texel_size;
    vec3 c = texture(src, v_uv).rgb * w0;
    c += texture(src, v_uv + step * 1.0).rgb * w1;
    c += texture(src, v_uv - step * 1.0).rgb * w1;
    c += texture(src, v_uv + step * 2.0).rgb * w2;
    c += texture(src, v_uv - step * 2.0).rgb * w2;
    c += texture(src, v_uv + step * 3.0).rgb * w3;
    c += texture(src, v_uv - step * 3.0).rgb * w3;
    c += texture(src, v_uv + step * 4.0).rgb * w4;
    c += texture(src, v_uv - step * 4.0).rgb * w4;
    frag_color = vec4(c, 1.0);
}
