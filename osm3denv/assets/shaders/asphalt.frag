#version 150

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 fog_colour;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in float v_fog_factor;

out vec4 frag_color;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p, int octaves) {
    float amp = 0.5;
    float sum = 0.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

void main() {
    vec2 uv = v_uv * 4.0;  // meters → texture-ish scale
    float coarse = fbm(uv * 0.5, 4);
    float fine = hash21(floor(uv * 40.0));
    float blotch = fbm(uv * 0.1, 3) * 0.5;

    float v = 0.22 + 0.10 * coarse + 0.04 * fine - 0.06 * blotch;
    // Sparkle: occasional bright pebble inclusions.
    float pebble_seed = hash21(floor(uv * 25.0));
    float pebble = smoothstep(0.94, 1.0, pebble_seed) * 0.25;
    v += pebble;
    v = clamp(v, 0.05, 0.5);

    vec3 base = vec3(v, v * 0.98, v * 1.02);

    vec3 N = normalize(v_world_normal);
    float diffuse = max(dot(N, -normalize(light_direction.xyz)), 0.0);
    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse);
    vec3 final = mix(fog_colour, lit, v_fog_factor);
    frag_color = vec4(final, 1.0);
}
