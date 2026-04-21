#version 150

// Final composite: add blurred bloom onto the HDR scene, apply ACES filmic
// tonemap, then gamma-correct to sRGB for display.

uniform sampler2D scene;
uniform sampler2D bloom;
uniform float bloom_strength;   // set by material, ~0.25

in vec2 v_uv;
out vec4 frag_color;

// ACES filmic tonemap — Krzysztof Narkowicz's cheap RRT+ODT approximation.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(scene, v_uv).rgb;
    vec3 blm = texture(bloom, v_uv).rgb;
    vec3 combined = hdr + blm * bloom_strength;
    vec3 ldr = aces(combined);
    // Output in sRGB: pow(·, 1/2.2) is a fast approximation. If the swap
    // chain is configured sRGB this should be removed — we keep it for
    // portability since Ogre's default viewport is linear-RGB.
    ldr = pow(ldr, vec3(1.0 / 2.2));
    frag_color = vec4(ldr, 1.0);
}
