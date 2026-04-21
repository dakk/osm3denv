#version 150

// PBR-textured terrain. The grass layer samples the ambientCG Grass004 pack.
// Rock, sand and snow keep the procedural generators from terrain.frag so we
// don't need all three PBR packs at once. Blending mirrors terrain.frag so
// transitions stay coherent with the procedural fallback.

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

uniform sampler2D grass_albedo;
uniform sampler2D grass_normal;
uniform sampler2D grass_rough;
uniform sampler2D grass_ao;
#ifdef PBR_ROCK_SAND
uniform sampler2D rock_albedo;
uniform sampler2D rock_normal;
uniform sampler2D rock_rough;
uniform sampler2D rock_ao;
uniform sampler2D sand_albedo;
uniform sampler2D sand_normal;
uniform sampler2D sand_rough;
uniform sampler2D sand_ao;
#endif

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

float detail_filter(vec2 p, float freq) {
    vec2 fw = fwidth(p) * freq;
    return 1.0 - smoothstep(0.5, 1.0, max(fw.x, fw.y));
}

float aa_hash_cell(vec2 p, float scale) {
    float aa = detail_filter(p, scale);
    return mix(0.5, hash21(floor(p * scale)), aa);
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

mat3 cotangent_frame(vec3 N, vec3 world_pos, vec2 uv) {
    vec3 dp1 = dFdx(world_pos);
    vec3 dp2 = dFdy(world_pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, N);
    vec3 dp1perp = cross(N, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));
    return mat3(T * invmax, B * invmax, N);
}

// ---------- Procedural rock / sand / snow ---------------------------------

vec3 rock_color(vec2 p, float slope_strength, float altitude, float detail) {
    float cell = worley(p * 0.7);
    float cracks = worley(p * 2.3);
    float grain = fbm(p * 1.4, 4);
    float micro = fbm(p * 5.0, 3) * detail_filter(p, 5.0);

    vec3 dark  = vec3(0.28, 0.26, 0.24);
    vec3 light = vec3(0.62, 0.58, 0.53);
    vec3 c = mix(dark, light, 1.0 - cell);
    c *= 0.82 + 0.30 * grain + 0.10 * micro;

    float edge = 1.0 - smoothstep(0.02, 0.12, cracks);
    c = mix(c, vec3(0.18, 0.17, 0.16), edge * 0.8);

    float lichen_mask = fbm(p * 0.35, 4);
    float lichen_w = smoothstep(0.55, 0.80, lichen_mask)
                   * (1.0 - smoothstep(0.55, 0.80, slope_strength))
                   * (1.0 - smoothstep(300.0, 900.0, altitude));
    c = mix(c, vec3(0.35, 0.48, 0.22), lichen_w * 0.55);

    float wet = 1.0 - smoothstep(-4.0, 6.0, altitude);
    c *= 1.0 - 0.35 * wet;

    float exposed = (1.0 - smoothstep(0.0, 0.4, slope_strength))
                  * smoothstep(500.0, 1500.0, altitude);
    c = mix(c, c * 1.15 + vec3(0.05), exposed * 0.5);
    return c;
}

vec3 sand_color(vec2 p, float detail) {
    vec2 dune_uv = vec2(p.x * 0.8 + p.y * 0.2, p.y);
    float dunes = sin(dune_uv.x * 0.9 + fbm(p * 0.18, 3) * 3.5) * 0.5 + 0.5;
    float fine = fbm(p * 1.7, 5);
    float ripple = fbm(p * 0.30, 3);
    float grain = aa_hash_cell(p, 14.0) * detail;

    vec3 pale = vec3(0.94, 0.84, 0.58);
    vec3 damp = vec3(0.70, 0.58, 0.36);
    vec3 c = mix(damp, pale, fine);
    c *= 0.88 + 0.14 * ripple + 0.10 * grain;
    c *= 0.85 + 0.25 * dunes;
    return c;
}

vec3 snow_color(vec2 p, float detail) {
    float drift = fbm(p * 0.6, 5);
    float sparkle = smoothstep(0.98, 1.0, hash21(floor(p * 40.0)))
                  * detail_filter(p, 40.0) * detail;
    vec3 warm = vec3(1.00, 0.98, 0.96);
    vec3 cool = vec3(0.78, 0.84, 0.92);
    vec3 c = mix(cool, warm, smoothstep(0.3, 0.7, drift));
    c += vec3(0.3) * sparkle;
    return c;
}

// ---------- Grass from PBR texture ---------------------------------------

struct GrassSample {
    vec3 albedo;
    vec3 n_tangent;     // tangent-space normal
    float roughness;
    float ao;
};

GrassSample sample_grass(vec2 uv_m) {
    // Pack represents ~1 m of lawn; sample once per metre.
    vec2 uv = uv_m;
    GrassSample g;
    g.albedo    = texture(grass_albedo, uv).rgb;
    g.n_tangent = texture(grass_normal, uv).rgb * 2.0 - 1.0;
    g.roughness = clamp(texture(grass_rough, uv).r, 0.10, 1.0);
    g.ao        = texture(grass_ao, uv).r;
    // Low-frequency albedo warp so the 1 m tile doesn't stripe the lawn.
    vec3 macro = texture(grass_albedo, uv * 0.09).rgb;
    float macro_lum = dot(macro, vec3(0.299, 0.587, 0.114));
    g.albedo *= mix(0.86, 1.16, macro_lum);
    return g;
}

// Bump from fbm gradient for the non-textured layers (rock, sand, snow).
vec3 procedural_bump(vec2 p, float detail) {
    float eps = 0.25;
    float coarse_aa = detail_filter(p, 0.8);
    float hc = fbm(p * 0.8, 4);
    float hx = fbm((p + vec2(eps, 0.0)) * 0.8, 4);
    float hy = fbm((p + vec2(0.0, eps)) * 0.8, 4);
    vec2 grad = vec2(hx - hc, hy - hc) / eps * coarse_aa;

    float fine_aa = detail_filter(p, 4.0);
    float fc = fbm(p * 4.0, 3);
    float fx = fbm((p + vec2(eps * 0.25, 0.0)) * 4.0, 3);
    float fy = fbm((p + vec2(0.0, eps * 0.25)) * 4.0, 3);
    vec2 fgrad = vec2(fx - fc, fy - fc) / (eps * 0.25);
    grad = grad + fgrad * fine_aa * detail * 0.5;

    return vec3(-grad.x, 0.0, -grad.y);
}

// ---------- Main -----------------------------------------------------------

void main() {
    vec3 N_geo = normalize(v_world_normal);
    float slope = 1.0 - clamp(dot(N_geo, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
    float altitude = v_world_pos.y;

    float cam_dist = length(v_world_pos - camera_position);
    float detail = 1.0 - smoothstep(80.0, 350.0, cam_dist);

    GrassSample g = sample_grass(v_uv);

    // Rock layer: PBR sampled when available, else procedural.
#ifdef PBR_ROCK_SAND
    vec2 rock_uv = v_uv * 0.5;   // 2 m per tile
    vec3 rock_col = texture(rock_albedo, rock_uv).rgb;
    float rock_rough = clamp(texture(rock_rough, rock_uv).r, 0.08, 1.0);
    float rock_ao_v = texture(rock_ao, rock_uv).r;
    vec3 rock_n_tan = texture(rock_normal, rock_uv).rgb * 2.0 - 1.0;
    // Macro luminance warp so 2 m tiles don't obviously repeat on cliffs.
    vec3 rock_macro = texture(rock_albedo, rock_uv * 0.12).rgb;
    rock_col *= mix(0.88, 1.12, dot(rock_macro, vec3(0.299, 0.587, 0.114)));
    // Carry over the procedural tints: wet-dark near sea, lichen on low
    // shallow slopes.
    float rock_wet = 1.0 - smoothstep(-4.0, 6.0, altitude);
    rock_col *= 1.0 - 0.25 * rock_wet;
    rock_rough = mix(rock_rough, rock_rough * 0.6, rock_wet);
    float rock_lichen = fbm(v_uv * 0.12, 4);
    float rock_lichen_w = smoothstep(0.55, 0.80, rock_lichen)
                        * (1.0 - smoothstep(0.55, 0.80, slope))
                        * (1.0 - smoothstep(300.0, 900.0, altitude));
    rock_col = mix(rock_col, vec3(0.35, 0.48, 0.22), rock_lichen_w * 0.45);
    vec3 rock_world_n = normalize(cotangent_frame(N_geo, v_world_pos, rock_uv)
                                  * rock_n_tan);
    vec3 rock = rock_col * rock_ao_v;
#else
    vec3 rock = rock_color(v_uv * 0.35, slope, altitude, detail);
    float rock_rough = 0.65;
    vec3 rock_world_n = N_geo;  // we'll add procedural bump later
#endif

    // Sand layer: PBR sampled when available, else procedural.
#ifdef PBR_ROCK_SAND
    vec2 sand_uv = v_uv * 0.5;
    vec3 sand_col = texture(sand_albedo, sand_uv).rgb;
    float sand_rough = clamp(texture(sand_rough, sand_uv).r, 0.08, 1.0);
    float sand_ao_v = texture(sand_ao, sand_uv).r;
    vec3 sand_n_tan = texture(sand_normal, sand_uv).rgb * 2.0 - 1.0;
    vec3 sand_macro = texture(sand_albedo, sand_uv * 0.10).rgb;
    sand_col *= mix(0.90, 1.10, dot(sand_macro, vec3(0.299, 0.587, 0.114)));
    vec3 sand_world_n = normalize(cotangent_frame(N_geo, v_world_pos, sand_uv)
                                  * sand_n_tan);
    vec3 sand = sand_col * sand_ao_v;
#else
    vec3 sand = sand_color(v_uv * 0.40, detail);
    float sand_rough = 0.94;
    vec3 sand_world_n = N_geo;
#endif

    vec3 snow = snow_color(v_uv * 0.60, detail);

    float w_snow  = smoothstep(0.0, 0.35, 1.0 - slope)
                  * smoothstep(1400.0, 1900.0, altitude);
    float w_rock  = smoothstep(0.22, 0.55, slope) * (1.0 - w_snow);
    float w_sand  = (1.0 - w_rock - w_snow) * smoothstep(2.0, -6.0, altitude);
    float w_grass = max(0.0, 1.0 - w_rock - w_sand - w_snow);
    float edge_noise = fbm(v_uv * 0.12, 3) - 0.5;
    w_rock  = clamp(w_rock  + edge_noise * 0.18, 0.0, 1.0);
    w_sand  = clamp(w_sand  + edge_noise * 0.14, 0.0, 1.0);
    w_snow  = clamp(w_snow  + edge_noise * 0.10, 0.0, 1.0);
    float wsum = w_grass + w_rock + w_sand + w_snow;
    w_grass /= wsum; w_rock /= wsum; w_sand /= wsum; w_snow /= wsum;

    // Albedo: blend grass texture with procedural rock/sand/snow.
    vec3 base = g.albedo * g.ao * w_grass
              + rock * w_rock
              + sand * w_sand
              + snow * w_snow;

    // Normal blend. Grass always uses its sampled normal via per-pixel TBN.
    // Rock/sand use their sampled normal when PBR is active, otherwise an
    // fbm-derived procedural bump. Snow stays on the geometric normal.
    mat3 TBN = cotangent_frame(N_geo, v_world_pos, v_uv);
    vec3 N_grass = normalize(TBN * g.n_tangent);
#ifdef PBR_ROCK_SAND
    vec3 N = normalize(N_grass * w_grass
                     + rock_world_n * w_rock
                     + sand_world_n * w_sand
                     + N_geo * w_snow);
#else
    float bump_strength = 0.35 * w_rock + 0.08 * w_sand + 0.04 * w_snow;
    vec3 bump = procedural_bump(v_uv, detail) * bump_strength;
    vec3 N = normalize(N_grass * w_grass + (N_geo + bump) * (1.0 - w_grass));
#endif

    // Roughness blend.
    float rough = g.roughness * w_grass
                + rock_rough * w_rock
                + sand_rough * w_sand
                + 0.55 * w_snow;
#ifndef PBR_ROCK_SAND
    // PBR rock branch already bakes the wet-smooth tweak into rock_rough.
    float wet = 1.0 - smoothstep(-4.0, 6.0, altitude);
    rough = mix(rough, rough * 0.55, wet * w_rock);
#endif

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(base, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           rough, 0.0);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
