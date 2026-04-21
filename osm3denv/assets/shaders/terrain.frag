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

// ---------- Anti-aliasing helpers ------------------------------------------
//
// Return a multiplier in [0, 1] that fades a feature of characteristic
// frequency ``freq`` when the on-screen pixel footprint grows larger than
// one period. Without this, hash-based sparkles and fine fbm octaves
// quantize to blocky pixel patterns at distance.

float detail_filter(vec2 p, float freq) {
    vec2 fw = fwidth(p) * freq;
    return 1.0 - smoothstep(0.5, 1.0, max(fw.x, fw.y));
}

// Anti-aliased hash cell: returns the hash value only when the pixel covers
// less than a full cell; otherwise returns the neutral 0.5 so it blends
// cleanly into neighbours. Use this anywhere we had ``hash21(floor(p*S))``.
float aa_hash_cell(vec2 p, float scale) {
    float aa = detail_filter(p, scale);
    return mix(0.5, hash21(floor(p * scale)), aa);
}

// ---------- Material generators --------------------------------------------
//
// Every generator takes a UV and a ``detail`` factor in [0, 1] so that
// distant fragments skip the finest octaves and avoid noise shimmer.

vec3 grass_color(vec2 p, float detail) {
    float base = fbm(p * 0.45, 5);
    float clump = fbm(p * 1.2, 4);
    // Fade blade-scale noise when pixels outgrow its period.
    float blade = fbm(p * 6.0, 3) * detail_filter(p, 6.0);
    float jitter = aa_hash_cell(p, 8.0);

    vec3 deep   = vec3(0.12, 0.26, 0.06);
    vec3 mid    = vec3(0.22, 0.44, 0.14);
    vec3 bright = vec3(0.36, 0.58, 0.20);
    vec3 dry    = vec3(0.58, 0.52, 0.22);
    vec3 moss   = vec3(0.16, 0.36, 0.12);

    vec3 c = mix(deep, mid, smoothstep(0.15, 0.75, base));
    c = mix(c, bright, smoothstep(0.55, 0.85, clump) * 0.55);
    c = mix(c, dry, smoothstep(0.72, 0.92, blade) * 0.7);
    c = mix(c, moss, smoothstep(0.02, 0.18, 1.0 - base) * 0.35);
    c *= 0.88 + 0.22 * jitter;

    // Wildflowers: high-frequency speckles, strictly gated by pixel footprint.
    float flower_aa = detail_filter(p, 26.0);
    float flower = hash21(floor(p * 26.0));
    float petal = smoothstep(0.985, 0.998, flower) * flower_aa * detail;
    vec3 bloom = mix(vec3(0.9, 0.8, 0.25), vec3(0.88, 0.32, 0.25),
                     aa_hash_cell(p, 13.0));
    c = mix(c, bloom, petal);
    return c;
}

vec3 rock_color(vec2 p, float slope_strength, float altitude,
                float detail) {
    float cell = worley(p * 0.7);
    float cracks = worley(p * 2.3);
    float grain = fbm(p * 1.4, 4);
    // Gate the fine octaves by pixel footprint: past ~20cm/pixel the micro
    // fbm is pure aliasing and should collapse to its mean.
    float micro = fbm(p * 5.0, 3) * detail_filter(p, 5.0);

    vec3 dark  = vec3(0.28, 0.26, 0.24);
    vec3 light = vec3(0.62, 0.58, 0.53);
    vec3 c = mix(dark, light, 1.0 - cell);
    c *= 0.82 + 0.30 * grain + 0.10 * micro;

    // Crack lines (narrow dark worley edges): darker + slightly blue-cool.
    float edge = 1.0 - smoothstep(0.02, 0.12, cracks);
    c = mix(c, vec3(0.18, 0.17, 0.16), edge * 0.8);

    // Lichen patches — soft green/yellow smudges only on gentle slopes
    // and at mid/low altitudes. Steep high cliffs stay clean.
    float lichen_mask = fbm(p * 0.35, 4);
    float lichen_w = smoothstep(0.55, 0.80, lichen_mask)
                   * (1.0 - smoothstep(0.55, 0.80, slope_strength))
                   * (1.0 - smoothstep(300.0, 900.0, altitude));
    vec3 lichen = vec3(0.35, 0.48, 0.22);
    c = mix(c, lichen, lichen_w * 0.55);

    // Wet/dark near sea level: moisture + spray makes low rock darker.
    float wet = 1.0 - smoothstep(-4.0, 6.0, altitude);
    c *= 1.0 - 0.35 * wet;

    // Exposed top: gentle slope + high altitude → slightly bleached.
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
    float sparkle_aa = detail_filter(p, 40.0);
    float sparkle = smoothstep(0.98, 1.0, hash21(floor(p * 40.0)))
                  * sparkle_aa * detail;
    vec3 warm = vec3(0.92, 0.92, 0.90);
    vec3 cool = vec3(0.70, 0.74, 0.82);
    vec3 c = mix(cool, warm, smoothstep(0.3, 0.7, drift));
    c += vec3(0.3) * sparkle;
    return c;
}

// Approximate terrain normal perturbation: take the fbm gradient on the
// world-XZ plane and tilt the geometric normal a little. Gives surfaces a
// lived-in bumpiness without touching the mesh. Each octave is gated by the
// pixel footprint so its bumps don't alias into shimmering speckle when the
// camera pulls back.
vec3 perturb_normal(vec3 N, vec2 p, float strength, float detail) {
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

    vec3 bump = vec3(-grad.x, 0.0, -grad.y) * strength;
    return normalize(N + bump);
}

// ---------- Main -----------------------------------------------------------

void main() {
    vec3 N_geo = normalize(v_world_normal);
    float slope = 1.0 - clamp(dot(N_geo, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
    float altitude = v_world_pos.y;

    // Distance-based detail fade: fine octaves phase out past ~300 m.
    float cam_dist = length(v_world_pos - camera_position);
    float detail = 1.0 - smoothstep(80.0, 350.0, cam_dist);

    vec2 uv_large = v_uv * 0.5;

    // Base material colors.
    vec3 grass = grass_color(uv_large, detail);
    vec3 rock  = rock_color (v_uv * 0.35, slope, altitude, detail);
    vec3 sand  = sand_color (v_uv * 0.40, detail);
    vec3 snow  = snow_color (v_uv * 0.60, detail);

    // Blend weights: snow on low-slope high altitude, rock on steep slopes,
    // sand near sea level, grass otherwise.
    float w_snow  = smoothstep(0.0, 0.20, 1.0 - slope)
                  * smoothstep(2400.0, 3200.0, altitude);
    float w_rock  = smoothstep(0.22, 0.55, slope) * (1.0 - w_snow);
    float w_sand  = (1.0 - w_rock - w_snow) * smoothstep(2.0, -6.0, altitude);
    float w_grass = max(0.0, 1.0 - w_rock - w_sand - w_snow);

    // Wobble the blend edges with FBM so the transitions aren't contour-line
    // smooth but look like natural patchwork.
    float edge_noise = fbm(v_uv * 0.12, 3) - 0.5;
    w_rock  = clamp(w_rock  + edge_noise * 0.18, 0.0, 1.0);
    w_sand  = clamp(w_sand  + edge_noise * 0.14, 0.0, 1.0);
    w_snow  = clamp(w_snow  + edge_noise * 0.10, 0.0, 1.0);
    float wsum = w_grass + w_rock + w_sand + w_snow;
    w_grass /= wsum; w_rock /= wsum; w_sand /= wsum; w_snow /= wsum;

    vec3 base = grass * w_grass + rock * w_rock
              + sand  * w_sand  + snow * w_snow;

    // Bump: strongest on rock, subtle on grass, very subtle on sand, almost
    // none on snow (smooth drifts).
    float bump_strength = 0.35 * w_rock + 0.12 * w_grass
                        + 0.08 * w_sand + 0.04 * w_snow;
    vec3 N = perturb_normal(N_geo, v_uv, bump_strength, detail);

    // Per-material roughness.
    float rough = 0.92 * w_grass + 0.65 * w_rock
                + 0.94 * w_sand  + 0.80 * w_snow;
    // Wet rock near the water line reads smoother.
    float wet = 1.0 - smoothstep(-4.0, 6.0, altitude);
    rough = mix(rough, rough * 0.55, wet * w_rock);

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(base, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           rough, 0.0);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
