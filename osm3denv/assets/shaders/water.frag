#version 150

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;
uniform vec3 fog_colour;
uniform float time;

in vec3 v_world_pos;
in vec2 v_uv_a;
in vec2 v_uv_b;
in float v_fog_factor;

out vec4 frag_color;

// Procedural ripple normal: superpose several directional sine waves,
// differentiate analytically to get a tangent-space normal.
vec3 ripple_normal(vec2 uv, float t) {
    vec2 dirs[4] = vec2[](
        normalize(vec2( 1.0,  0.3)),
        normalize(vec2(-0.6,  1.0)),
        normalize(vec2( 0.7, -0.8)),
        normalize(vec2(-1.0, -0.2))
    );
    float freqs[4] = float[](6.0, 11.0, 18.0, 24.0);
    float amps[4]  = float[](0.10, 0.06, 0.03, 0.015);
    float speeds[4] = float[](1.0, 0.7, 1.4, 1.9);

    float dhdx = 0.0;
    float dhdy = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 d = dirs[i];
        float phase = dot(d, uv) * freqs[i] - t * speeds[i];
        float k = amps[i] * freqs[i];
        float c = cos(phase) * k;
        dhdx += c * d.x;
        dhdy += c * d.y;
    }
    vec3 n = normalize(vec3(-dhdx, 1.5, -dhdy));
    return n;
}

void main() {
    // Two offset samples at different scales so the pattern doesn't feel grid-aligned.
    vec3 n1 = ripple_normal(v_uv_a * 8.0, time * 0.8);
    vec3 n2 = ripple_normal(v_uv_b * 5.0, time * 0.5);
    vec3 N = normalize(n1 + n2);

    vec3 base = vec3(0.10, 0.28, 0.46);

    vec3 L = -normalize(light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 H = normalize(L + V);
    float diffuse = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse)
             + vec3(1.0) * spec * 0.8;
    vec3 final = mix(fog_colour, lit, v_fog_factor);
    frag_color = vec4(final, 0.82);
}
