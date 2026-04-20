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

// ---------- Noise helpers --------------------------------------------------

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    return vec2(hash21(p), hash21(p + 17.31));
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
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

float worley(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float d = 1.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neigh = vec2(x, y);
            vec2 feat = neigh + hash22(i + neigh);
            d = min(d, length(feat - f));
        }
    }
    return d;
}

// ---------- Material generators --------------------------------------------

vec3 grass_color(vec2 p) {
    float base = fbm(p * 0.45, 5);
    float blade = fbm(p * 6.0, 3);
    float jitter = hash21(floor(p * 8.0));
    vec3 dark = vec3(0.14, 0.30, 0.08);
    vec3 light = vec3(0.32, 0.55, 0.18);
    vec3 dry = vec3(0.52, 0.48, 0.20);
    vec3 c = mix(dark, light, smoothstep(0.2, 0.9, base));
    c = mix(c, dry, smoothstep(0.75, 0.95, blade) * 0.7);
    c *= 0.85 + 0.30 * jitter;
    return c;
}

vec3 rock_color(vec2 p) {
    float cell = worley(p * 0.7);
    float grain = fbm(p * 1.4, 4);
    vec3 dark = vec3(0.30, 0.28, 0.25);
    vec3 light = vec3(0.62, 0.58, 0.53);
    vec3 c = mix(dark, light, 1.0 - cell);
    c *= 0.85 + 0.30 * grain;
    float crack = 1.0 - smoothstep(0.35, 0.55, cell);
    c *= 1.0 - crack * 0.5;
    return c;
}

vec3 sand_color(vec2 p) {
    float fine = fbm(p * 1.7, 5);
    float ripple = fbm(p * 0.30, 3);
    float grain = hash21(floor(p * 14.0));
    vec3 pale = vec3(0.92, 0.82, 0.56);
    vec3 damp = vec3(0.72, 0.60, 0.38);
    vec3 c = mix(damp, pale, fine);
    c *= 0.90 + 0.18 * ripple + 0.10 * grain;
    return c;
}

// ---------- Main -----------------------------------------------------------

void main() {
    vec3 N = normalize(v_world_normal);
    float slope = 1.0 - clamp(dot(N, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
    float altitude = v_world_pos.y;

    vec3 grass = grass_color(v_uv * 0.5);
    vec3 rock = rock_color(v_uv * 0.35);
    vec3 sand = sand_color(v_uv * 0.40);

    float w_rock = smoothstep(0.22, 0.55, slope);
    float w_sand = (1.0 - w_rock) * smoothstep(2.0, -6.0, altitude);
    float w_grass = max(0.0, 1.0 - w_rock - w_sand);
    vec3 base = grass * w_grass + rock * w_rock + sand * w_sand;

    float diffuse = max(dot(N, -normalize(light_direction.xyz)), 0.0);
    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse);
    vec3 final = mix(fog_colour, lit, v_fog_factor);
    frag_color = vec4(final, 1.0);
}
