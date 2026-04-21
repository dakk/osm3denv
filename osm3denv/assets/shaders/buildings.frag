#version 150

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

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
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kd * albedo / PBR_PI;
    vec3 direct = (diff + spec) * ndl * sun_color;

    vec3 R = reflect(-V, N);
    vec3 env_diff = atmos_sky(N, L) * albedo * (1.0 - metallic) * 0.35;
    vec3 env_spec = atmos_sky(normalize(R), L)
                  * pbr_F_schlick(ndv, F0)
                  * (1.0 - roughness * 0.9) * 0.35;
    vec3 floor_amb = albedo * ambient_col * (1.0 - metallic) * 0.5;

    return direct + env_diff + env_spec + floor_amb;
}

// Standard brick: ~25 cm wide, 7 cm tall, with half-offset alternating rows.
vec3 brick_wall(vec2 uv) {
    vec2 brick = vec2(0.25, 0.07);
    float row_idx = floor(uv.y / brick.y);
    float row_offset = mod(row_idx, 2.0) * brick.x * 0.5;
    vec2 local = vec2(uv.x - row_offset, uv.y);
    vec2 cell = floor(local / brick);
    vec2 f = fract(local / brick);

    float mortar_frac = 0.10;
    float m = 1.0 -
        smoothstep(0.0, mortar_frac, f.x) *
        smoothstep(0.0, mortar_frac, 1.0 - f.x) *
        smoothstep(0.0, mortar_frac, f.y) *
        smoothstep(0.0, mortar_frac, 1.0 - f.y);

    float seed = hash21(cell + row_offset);
    vec3 brick_cols[6] = vec3[](
        vec3(0.62, 0.24, 0.15),
        vec3(0.55, 0.22, 0.15),
        vec3(0.58, 0.28, 0.18),
        vec3(0.50, 0.20, 0.13),
        vec3(0.68, 0.30, 0.20),
        vec3(0.48, 0.22, 0.17)
    );
    int pick = int(mod(floor(seed * 6.0), 6.0));
    vec3 bc = brick_cols[pick];
    bc *= 0.82 + 0.30 * fbm(uv * 40.0, 3);
    float edge = 1.0 - 0.35 * max(max(abs(f.x - 0.5) * 2.0, abs(f.y - 0.5) * 2.0), 0.0);
    bc *= edge;

    vec3 mortar_col = vec3(0.42, 0.40, 0.37) * (0.9 + 0.2 * hash21(cell * 3.7));
    return mix(bc, mortar_col, m);
}

// Emits ``glass`` in [0, 1] = 1 where the fragment is a glass pane (not
// frame, not wall). Callers use this to bump glossiness for window panes.
vec3 windows(vec2 uv, vec3 wall_col, out float glass) {
    vec2 spacing = vec2(1.6, 3.0);
    vec2 size = vec2(0.9, 1.4);
    vec2 cell = floor(uv / spacing);
    vec2 within = fract(uv / spacing) * spacing - (spacing - size) * 0.5;
    float alive = step(0.5, cell.y);
    bvec2 inside = bvec2(within.x > 0.0 && within.x < size.x,
                          within.y > 0.0 && within.y < size.y);
    glass = 0.0;
    if (alive > 0.5 && inside.x && inside.y) {
        float ws = hash21(cell);
        vec3 glass_col = mix(vec3(0.16, 0.20, 0.28), vec3(0.30, 0.38, 0.48), ws);
        vec2 d_edge = min(within, size - within);
        float frame = 1.0 - smoothstep(0.0, 0.05, min(d_edge.x, d_edge.y));
        vec3 frame_col = vec3(0.15, 0.12, 0.10);
        glass = 1.0 - frame;   // pane only, frame stays matte
        return mix(glass_col, frame_col, frame);
    }
    return wall_col;
}

vec3 roof_pattern(vec2 uv) {
    float row = uv.y / 0.30;
    float band = abs(fract(row) - 0.5) * 2.0;
    float shade = mix(0.55, 1.0, 1.0 - band);
    float col_jit = hash21(vec2(floor(uv.x / 0.22), floor(row)));
    vec3 tile_base = mix(vec3(0.58, 0.30, 0.20), vec3(0.76, 0.42, 0.25), col_jit);
    vec3 groove = vec3(0.35, 0.25, 0.18);
    vec3 c = mix(groove, tile_base, shade);
    c *= 0.90 + 0.15 * fbm(uv * 8.0, 3);
    return c;
}

void main() {
    vec3 N = normalize(v_world_normal);
    float up = N.y;
    vec3 base;
    float roughness;
    float metallic = 0.0;
    if (up > 0.6) {
        base = roof_pattern(v_uv);
        roughness = 0.75;                   // terracotta tiles
    } else if (up < -0.6) {
        base = vec3(0.28, 0.26, 0.24);
        roughness = 0.90;                   // unlit underside, matte
    } else {
        base = brick_wall(v_uv);
        float glass;
        base = windows(v_uv, base, glass);
        // Brick matte (0.88) → glass glossy (0.12). Frames stay matte via glass=0.
        roughness = mix(0.88, 0.12, glass);
    }

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(base, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           roughness, metallic);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
