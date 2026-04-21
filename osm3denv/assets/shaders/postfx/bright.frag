#version 150

// Bloom extract: forward pixels whose luminance is above the threshold, with
// a soft falloff so highlights don't hard-edge. The scene buffer is HDR so
// the sun disk, window specular glints and water highlights will surface.

uniform sampler2D scene;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 c = texture(scene, v_uv).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    // Soft knee around lum = 1.0: below 0.9 contributes nothing, above 1.4
    // passes through fully. Adjust for more/less bloom.
    float bright = smoothstep(0.9, 1.4, lum);
    frag_color = vec4(c * bright, 1.0);
}
