#version 150

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
    float aerial = 1.0 - exp(-max(d - 100.0, 0.0) * 0.0004);
    return mix(lit, atmos_sky(view_dir, sun_dir), aerial);
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

    vec3 sun_dir = normalize(-light_direction.xyz);
    float diffuse = max(dot(N, sun_dir), 0.0);
    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
