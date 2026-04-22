#version 150

// PBR road shader with procedural lane markings. ``surface_albedo`` etc.
// sample whatever pack the material binds (asphalt, paving stones, dirt,
// gravel, etc.). ``marking_mode`` selects which (if any) lane markings to
// overlay on top of the sampled surface:
//   0 = none
//   1 = dashed white centreline (minor roads)
//   2 = solid double-yellow centreline (major roads)
//
// The vertex UV is road-local: uv.x is metres from the centreline,
// uv.y is cumulative metres along the road. Lane-marking thresholds are in
// those units so the markings have correct real-world dimensions.

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

uniform sampler2D surface_albedo;
uniform sampler2D surface_normal;
uniform sampler2D surface_rough;
uniform sampler2D surface_ao;

uniform int marking_mode;
uniform float edge_offset;    // half-width (m) for solid edge lines; 0 = off

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

out vec4 frag_color;

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
    vec3 F = pbr_F_schlick(vdh, F0);
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

// ---------- Lane markings --------------------------------------------------
//
// All widths are in metres because the vertex UV is in metres (lateral from
// centreline × distance along road). ``u`` = v_uv.x, ``v`` = v_uv.y.
//
// Soft-edge helper so marking borders anti-alias with pixel footprint.

float soft_band(float x, float half_width) {
    // 1 inside [-half, +half], 0 outside, with a pixel-sized transition.
    float ax = abs(x);
    float aa = fwidth(x);
    return 1.0 - smoothstep(half_width - aa, half_width + aa, ax);
}

vec4 compute_marking(vec2 uv) {
    // Returns (rgb_color, intensity). intensity = 0 → no marking.
    vec4 none = vec4(0.0);
    if (marking_mode == 0) return none;

    float u = uv.x;
    float v = uv.y;

    // Solid white edge lines just inside each kerb for any road that has
    // markings. The road ribbon UV.x spans [-half_width, +half_width], and
    // ``edge_offset`` is the half-width per material; paint an 0.12 m line
    // 8 cm inside that.
    if (edge_offset > 0.1) {
        float line_half = 0.06;
        float inset = 0.08;
        float left_edge  = abs(u + (edge_offset - inset));
        float right_edge = abs(u - (edge_offset - inset));
        float edge = min(left_edge, right_edge);
        float aa = fwidth(u);
        float line = 1.0 - smoothstep(line_half - aa, line_half + aa, edge);
        if (line > 0.0) return vec4(vec3(1.0), line);
    }

    if (marking_mode == 1) {
        // Dashed white centreline: 3 m painted line, 9 m gap.
        float line_half = 0.08;   // 16 cm wide line
        float lateral = soft_band(u, line_half);
        if (lateral <= 0.0) return none;
        float cycle = mod(v, 12.0);
        float dash = step(cycle, 3.0);
        return vec4(vec3(1.0), lateral * dash);
    }

    if (marking_mode == 2) {
        // Solid double yellow in the middle, plus dashed white lane divider.
        // Double yellow: two 12 cm lines, 12 cm gap, centred on u=0.
        float yellow_half = 0.06;
        float yellow_offset = 0.09;
        float left_yellow  = soft_band(u + yellow_offset, yellow_half);
        float right_yellow = soft_band(u - yellow_offset, yellow_half);
        float yellow = max(left_yellow, right_yellow);
        if (yellow > 0.0) return vec4(vec3(0.95, 0.78, 0.20), yellow);

        // White dashed lane dividers at ±3.5 m (two-lane motorway
        // approximation). 3 m line, 9 m gap.
        float divider_offset = 3.5;
        float divider_half = 0.06;
        float left_div  = soft_band(u + divider_offset, divider_half);
        float right_div = soft_band(u - divider_offset, divider_half);
        float div_lat = max(left_div, right_div);
        if (div_lat > 0.0) {
            float cycle = mod(v, 12.0);
            float dash = step(cycle, 3.0);
            return vec4(vec3(1.0), div_lat * dash);
        }
    }
    return none;
}

// ---------- Main -----------------------------------------------------------

void main() {
    // The asphalt/paving pack is 2 m square; road-local UV is already in
    // metres, so dividing by 2 gives one tile per 2 m. For texture sampling
    // we also want the texture to run ALONG the road (v axis) rather than
    // across it; our UV already does that because v increases with distance.
    vec2 tex_uv = v_uv * 0.5;

    vec3 albedo = texture(surface_albedo, tex_uv).rgb;
    float rough = clamp(texture(surface_rough, tex_uv).r, 0.08, 1.0);
    float ao    = texture(surface_ao, tex_uv).r;
    vec3 n_tan  = texture(surface_normal, tex_uv).rgb * 2.0 - 1.0;

    // Anti-tile macro warp.
    vec3 macro = texture(surface_albedo, tex_uv * 0.15).rgb;
    float macro_lum = dot(macro, vec3(0.299, 0.587, 0.114));
    albedo *= mix(0.88, 1.12, macro_lum);

    vec3 N_geo = normalize(v_world_normal);
    mat3 TBN = cotangent_frame(N_geo, v_world_pos, tex_uv);
    vec3 N = normalize(TBN * n_tan);

    // Composite lane markings on top of the sampled surface.
    vec4 mark = compute_marking(v_uv);
    if (mark.a > 0.0) {
        albedo = mix(albedo, mark.rgb, mark.a);
        // Paint is brighter (lower AO) and slightly smoother than asphalt.
        rough = mix(rough, 0.55, mark.a);
        ao = mix(ao, 1.0, mark.a);
        N = mix(N, N_geo, mark.a);   // paint hides the asphalt bumps
    }

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(albedo * ao, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           rough, 0.0);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
