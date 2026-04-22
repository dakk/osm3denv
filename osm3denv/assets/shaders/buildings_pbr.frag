#version 150

// Textured PBR buildings. Walls sample the ambientCG Bricks097 pack; roof
// and underside stay procedural (roof tiles + windows overlay are too
// content-specific for a generic brick atlas). Windows are composited on top
// of the sampled brick exactly as in the procedural shader.

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

uniform sampler2D brick_albedo;
uniform sampler2D brick_normal;
uniform sampler2D brick_rough;
uniform sampler2D brick_ao;
#ifdef PBR_ROOF
uniform sampler2D roof_albedo;
uniform sampler2D roof_normal;
uniform sampler2D roof_rough;
uniform sampler2D roof_ao;
#endif

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in vec4 v_color;     // per-building palette tint

out vec4 frag_color;

// ---------- Noise helpers (procedural roof + windows still need them) ------

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

// ---------- Procedural roof tiles + windows (kept from buildings.frag) -----

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

// Procedural window overlay: window opening + frame + flanking shutters.
// Returns the composited wall colour and sets ``glass`` ≥ 0 where the
// fragment is inside an actual glass pane (so the caller can bump its
// roughness / AO). Ground-floor cells (cell.y == 0) are left empty so the
// door overlay can claim them in main().
vec3 windows(vec2 uv, vec3 wall_col, out float glass) {
    vec2 spacing = vec2(1.6, 3.0);
    vec2 size = vec2(0.9, 1.4);
    vec2 shutter_size = vec2(0.22, 1.4);
    float shutter_gap = 0.04;
    vec2 cell = floor(uv / spacing);
    vec2 within = fract(uv / spacing) * spacing - spacing * 0.5; // centred
    glass = 0.0;
    float alive = step(0.5, cell.y);
    if (alive <= 0.5) return wall_col;

    // --- Window opening (centred on (0, 0)) ---------------------------
    vec2 d_win = abs(within) - size * 0.5;
    if (d_win.x < 0.0 && d_win.y < 0.0) {
        float ws = hash21(cell);
        vec3 glass_col = mix(vec3(0.16, 0.20, 0.28), vec3(0.30, 0.38, 0.48), ws);
        // Outer frame + cross mullions (one vertical, one horizontal) that
        // divide the pane into quarters like a real casement window.
        float frame = 1.0 - smoothstep(0.0, 0.05, min(-d_win.x, -d_win.y));
        float mull_v = 1.0 - smoothstep(0.015, 0.030, abs(within.x));
        float mull_h = 1.0 - smoothstep(0.015, 0.030, abs(within.y));
        float all_frame = max(frame, max(mull_v, mull_h));
        vec3 frame_col = vec3(0.15, 0.12, 0.10);
        glass = 1.0 - all_frame;
        return mix(glass_col, frame_col, all_frame);
    }

    // --- Stone lintel above the window -------------------------------
    if (within.y > size.y * 0.5 + 0.03 && within.y < size.y * 0.5 + 0.11 &&
        abs(within.x) < size.x * 0.5 + 0.08) {
        return vec3(0.84, 0.80, 0.72);
    }
    // --- Stone sill just below the window ----------------------------
    if (within.y < -size.y * 0.5 - 0.02 && within.y > -size.y * 0.5 - 0.12 &&
        abs(within.x) < size.x * 0.5 + 0.10) {
        return vec3(0.84, 0.80, 0.72);
    }

    // --- Shutters flanking the window --------------------------------
    // Both shutters occupy the same vertical range as the window opening.
    if (abs(within.y) < size.y * 0.5) {
        float sh_inner = size.x * 0.5 + shutter_gap;
        float sh_outer = sh_inner + shutter_size.x;
        if (abs(within.x) > sh_inner && abs(within.x) < sh_outer) {
            // Per-cell shutter colour: Mediterranean dark-green, brown or
            // blue-grey. Louvre pattern adds subtle horizontal lines.
            float sh_seed = hash21(cell + vec2(13.7, 5.1));
            vec3 col_a = vec3(0.22, 0.35, 0.22);   // forest green
            vec3 col_b = vec3(0.32, 0.20, 0.12);   // dark walnut
            vec3 col_c = vec3(0.20, 0.26, 0.32);   // slate blue
            vec3 shutter_col = mix(col_a, col_b,
                                   step(0.33, sh_seed) - step(0.66, sh_seed));
            shutter_col = mix(shutter_col, col_c, step(0.66, sh_seed));
            float louvre = fract(within.y * 10.0);
            shutter_col *= mix(0.78, 1.05,
                               smoothstep(0.25, 0.75, louvre));
            return shutter_col;
        }
    }
    return wall_col;
}

// Procedural front door. Returns wall_col untouched unless the fragment
// lies inside the door rectangle on the designated door edge, indicated by
// a non-zero v_color.a carrying the door-centre running-length UV position.
vec3 door(vec2 uv, vec3 wall_col, float door_u) {
    if (door_u < 0.01) return wall_col;
    float door_half_w = 0.55;    // 1.1 m wide
    float door_h = 2.10;         // 2.1 m tall
    float dx = uv.x - door_u;
    if (abs(dx) < door_half_w && uv.y < door_h) {
        vec2 d_local = vec2(abs(dx), uv.y);
        // Dark wood door with brass-ish handle notch.
        vec3 door_col = vec3(0.22, 0.14, 0.08);
        // Panel lines at door/2 height + horizontal rule.
        float panel_v = smoothstep(0.02, 0.0,
                                   abs(d_local.y - door_h * 0.5));
        float panel_h = smoothstep(0.02, 0.0,
                                   abs(d_local.x - door_half_w * 0.5));
        door_col = mix(door_col, door_col * 0.6, max(panel_v, panel_h) * 0.8);
        // Frame around the opening.
        float frame_d = min(door_half_w - abs(dx),
                            min(uv.y, door_h - uv.y));
        float frame = 1.0 - smoothstep(0.0, 0.06, frame_d);
        door_col = mix(door_col, vec3(0.10, 0.08, 0.06), frame);
        return door_col;
    }
    return wall_col;
}

// ---------- Main -----------------------------------------------------------

void main() {
    vec3 N_geo = normalize(v_world_normal);
    float up = N_geo.y;

    vec3 albedo;
    vec3 N = N_geo;
    float roughness;
    float ao = 1.0;
    float metallic = 0.0;

    if (up > 0.35) {
#ifdef PBR_ROOF
        // Roof: sample the ambientCG roofing tile pack. Pack represents ~1 m
        // square, so uv directly gives the right tile rate.
        vec2 uv = v_uv;
        albedo = texture(roof_albedo, uv).rgb;
        roughness = clamp(texture(roof_rough, uv).r, 0.08, 1.0);
        ao = texture(roof_ao, uv).r;
        vec3 n_tangent = texture(roof_normal, uv).rgb * 2.0 - 1.0;
        mat3 TBN = cotangent_frame(N_geo, v_world_pos, uv);
        N = normalize(TBN * n_tangent);
        vec3 macro = texture(roof_albedo, uv * 0.13).rgb;
        float macro_lum = dot(macro, vec3(0.299, 0.587, 0.114));
        albedo *= mix(0.88, 1.12, macro_lum);
#else
        albedo = roof_pattern(v_uv);
        roughness = 0.75;
#endif
    } else if (up < -0.35) {
        albedo = vec3(0.28, 0.26, 0.24);
        roughness = 0.90;
    } else {
        // Wall: sample the brick PBR pack. UVs are in metres; the pack
        // represents a ~1 m patch, so use uv directly.
        vec2 uv = v_uv;
        albedo = texture(brick_albedo, uv).rgb;
        roughness = clamp(texture(brick_rough, uv).r, 0.08, 1.0);
        ao = texture(brick_ao, uv).r;
        vec3 n_tangent = texture(brick_normal, uv).rgb * 2.0 - 1.0;
        mat3 TBN = cotangent_frame(N_geo, v_world_pos, uv);
        N = normalize(TBN * n_tangent);

        // Low-frequency albedo warp to break up repetition across big walls.
        vec3 macro = texture(brick_albedo, uv * 0.13).rgb;
        float macro_lum = dot(macro, vec3(0.299, 0.587, 0.114));
        albedo *= mix(0.88, 1.12, macro_lum);

        // Per-building palette tint applied to the sampled wall before
        // windows are composited, so glass panes stay uncoloured.
        albedo *= v_color.rgb;

        // Overlay procedural windows on the sampled brick.
        float glass;
        albedo = windows(v_uv, albedo, glass);
        if (glass > 0.0) {
            // Glass panes: flat, glossy, high AO (no gap).
            roughness = mix(roughness, 0.12, glass);
            ao = mix(ao, 1.0, glass);
            N = mix(N, N_geo, glass);
        }
        // Front door on the designated wall edge (v_color.a carries its
        // running-length UV centre; 0 on non-door edges).
        vec3 pre_door = albedo;
        albedo = door(v_uv, albedo, v_color.a);
        if (albedo != pre_door) {
            roughness = 0.60;
            N = N_geo;
        }
    }

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(albedo * ao, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           roughness, metallic);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
