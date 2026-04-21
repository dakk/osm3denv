#version 150

// Single fragment shader for all draped "area" layers (vegetation, landuse,
// natural surfaces). Each material sets ``area_class`` to pick the look.
//   1 = vegetation   2 = residential   3 = commercial   4 = industrial
//   5 = farmland     6 = sand          7 = rock

uniform int area_class;

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

out vec4 frag_color;

// ---------- Noise helpers --------------------------------------------------

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

// ---------- Atmospheric scatter (shared with sky.frag) ---------------------

vec3 atmos_sky(vec3 v, vec3 s) {
    float h = v.y;
    float sh = s.y;
    float mu = clamp(dot(v, s), -1.0, 1.0);
    vec3 zenith  = vec3(0.22, 0.48, 0.85);
    vec3 horizon = vec3(0.75, 0.85, 0.92);
    vec3 day_col = mix(horizon, zenith, smoothstep(0.0, 0.6, max(h, 0.0)));
    float ang = acos(mu);
    float halo = exp(-ang * 6.0);
    vec3 halo_tint = vec3(1.00, 0.88, 0.65);
    day_col = mix(day_col, day_col + halo_tint * 0.5, halo * 0.6);
    float g = 0.82;
    float phase_m = (1.0 - g*g) / pow(max(1.0 + g*g - 2.0*g*mu, 1e-4), 1.5);
    day_col += halo_tint * phase_m * 0.005;
    float sun_disk = smoothstep(0.9996, 0.9999, mu);
    day_col = mix(day_col, vec3(1.40, 1.25, 1.00), sun_disk);
    float dusk = smoothstep(0.25, -0.05, sh);
    float near_h = 1.0 - smoothstep(0.0, 0.35, max(h, 0.0));
    vec3 dusk_tint = vec3(1.10, 0.55, 0.25);
    day_col = mix(day_col, dusk_tint, dusk * near_h * 0.75);
    float day = smoothstep(-0.10, 0.20, sh);
    vec3 night = vec3(0.02, 0.03, 0.06);
    vec3 C = mix(night, day_col, day);
    C *= smoothstep(-0.20, 0.0, h) * 0.55 + 0.45;
    return C;
}

vec3 apply_aerial(vec3 lit, vec3 world_pos, vec3 cam_pos, vec3 sun_dir) {
    vec3 v = world_pos - cam_pos;
    float d = length(v);
    vec3 view_dir = v / max(d, 1e-4);
    float aerial = (1.0 - exp(-max(d - 200.0, 0.0) * 0.00020)) * 0.50;
    return mix(lit, atmos_sky(view_dir, sun_dir), aerial);
}

// ---------- PBR (analytical Cook-Torrance + GGX) ---------------------------

const float PBR_PI = 3.14159265;

float pbr_D_ggx(float ndh, float alpha) {
    float a2 = alpha * alpha;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PBR_PI * d * d);
}

float pbr_G1(float nv, float alpha) {
    float k = (alpha + 1.0);
    k = k * k * 0.125;
    return nv / (nv * (1.0 - k) + k);
}

vec3 pbr_F_schlick(float cos_t, vec3 f0) {
    float x = 1.0 - clamp(cos_t, 0.0, 1.0);
    float x5 = x * x * x * x * x;
    return f0 + (1.0 - f0) * x5;
}

vec3 pbr_surface(vec3 albedo, vec3 N, vec3 V, vec3 L,
                 vec3 sun_color, vec3 ambient_col,
                 float roughness, float metallic) {
    vec3 H = normalize(V + L);
    float ndl = max(dot(N, L), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float ndh = max(dot(N, H), 0.0);
    float vdh = max(dot(V, H), 0.0);
    float alpha = roughness * roughness;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float D = pbr_D_ggx(ndh, alpha);
    float G = pbr_G1(ndv, alpha) * pbr_G1(ndl, alpha);
    vec3  F = pbr_F_schlick(vdh, F0);

    vec3 spec = (D * G * F) / max(4.0 * ndv * ndl, 1e-4);
    // Rough dielectrics physically produce negligible specular peaks.
    // Attenuate the direct term quadratically so grass/rock/sand don't
    // flash a bright hotspot where N*H happens to align with the sun.
    float _spec_fade = 1.0 - roughness;
    spec *= _spec_fade * _spec_fade;
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kd * albedo / PBR_PI;
    vec3 direct = (diff + spec) * ndl * sun_color;

    vec3 R = reflect(-V, N);
    vec3 env_diff = atmos_sky(N, L) * albedo * (1.0 - metallic) * 0.35;
    vec3 env_spec = atmos_sky(normalize(R), L)
                  * pbr_F_schlick(ndv, F0)
                  * max(0.0, 1.0 - roughness * 1.3) * 0.25;
    vec3 floor_amb = albedo * ambient_col * (1.0 - metallic) * 0.5;

    return direct + env_diff + env_spec + floor_amb;
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
    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    // Roughness per class: paved commercial/industrial a hair smoother than
    // vegetation/farmland/sand/rock. All dielectric.
    float roughness = 0.90;
    if      (area_class == 3 || area_class == 4) roughness = 0.75;
    else if (area_class == 2)                    roughness = 0.82;
    else if (area_class == 7)                    roughness = 0.72;
    vec3 lit = pbr_surface(base, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           roughness, 0.0);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
