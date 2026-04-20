#version 150

// Single fragment shader for all draped "area" layers (vegetation, landuse,
// natural surfaces). Each material sets ``area_class`` to pick the look.
//   1 = vegetation   2 = residential   3 = commercial   4 = industrial
//   5 = farmland     6 = sand          7 = rock

uniform int area_class;

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 fog_colour;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in float v_fog_factor;

out vec4 frag_color;

// ---------- Noise helpers (identical to terrain.frag) ----------------------

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) { return vec2(hash21(p), hash21(p + 17.31)); }

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

// ---------- Per-class procedural colours -----------------------------------

vec3 vegetation_color(vec2 p) {
    float base = fbm(p * 0.25, 5);
    float clump = fbm(p * 1.2, 4);
    vec3 deep = vec3(0.10, 0.25, 0.06);
    vec3 leaf = vec3(0.20, 0.42, 0.14);
    vec3 bright = vec3(0.32, 0.55, 0.20);
    vec3 c = mix(deep, leaf, base);
    c = mix(c, bright, smoothstep(0.55, 0.85, clump) * 0.6);
    c *= 0.85 + 0.25 * hash21(floor(p * 10.0));
    return c;
}

vec3 residential_color(vec2 p) {
    float patches = fbm(p * 0.3, 4);
    vec3 gravel = vec3(0.72, 0.66, 0.55);
    vec3 grass = vec3(0.30, 0.48, 0.22);
    vec3 pave = vec3(0.55, 0.52, 0.48);
    vec3 c = mix(gravel, pave, smoothstep(0.3, 0.7, patches));
    c = mix(c, grass, smoothstep(0.6, 0.8, fbm(p * 0.8, 3)) * 0.35);
    c *= 0.90 + 0.15 * hash21(floor(p * 6.0));
    return c;
}

vec3 commercial_color(vec2 p) {
    float pave = fbm(p * 0.5, 4);
    vec3 c = mix(vec3(0.55, 0.46, 0.36), vec3(0.72, 0.60, 0.48), pave);
    c *= 0.90 + 0.15 * fbm(p * 4.0, 2);
    return c;
}

vec3 industrial_color(vec2 p) {
    float slow = fbm(p * 0.3, 4);
    float stain = worley(p * 0.6);
    vec3 base = mix(vec3(0.42, 0.42, 0.42), vec3(0.58, 0.58, 0.58), slow);
    base *= 1.0 - smoothstep(0.7, 0.95, 1.0 - stain) * 0.35;
    base *= 0.90 + 0.10 * fbm(p * 5.0, 2);
    return base;
}

vec3 farmland_color(vec2 p) {
    float stripes = sin(p.y * 2.0) * 0.5 + 0.5;
    float noise = fbm(p * 0.4, 4);
    vec3 earth = vec3(0.45, 0.32, 0.18);
    vec3 crop = vec3(0.62, 0.58, 0.28);
    vec3 c = mix(earth, crop, smoothstep(0.3, 0.8, stripes * 0.7 + noise * 0.5));
    c *= 0.88 + 0.18 * hash21(floor(p * 12.0));
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

// ---------- Main -----------------------------------------------------------

void main() {
    vec2 p = v_uv;
    vec3 base;
    if      (area_class == 1) base = vegetation_color(p * 0.5);
    else if (area_class == 2) base = residential_color(p * 0.25);
    else if (area_class == 3) base = commercial_color(p * 0.25);
    else if (area_class == 4) base = industrial_color(p * 0.25);
    else if (area_class == 5) base = farmland_color(p * 0.40);
    else if (area_class == 6) base = sand_color(p * 0.40);
    else if (area_class == 7) base = rock_color(p * 0.35);
    else                       base = vec3(0.7, 0.7, 0.7);

    vec3 N = normalize(v_world_normal);
    float diffuse = max(dot(N, -normalize(light_direction.xyz)), 0.0);
    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse);
    vec3 final = mix(fog_colour, lit, v_fog_factor);
    frag_color = vec4(final, 1.0);
}
