#version 150

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;
uniform float time;

in vec3 v_world_pos;
in vec2 v_uv_a;
in vec2 v_uv_b;

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
    float aerial = 1.0 - exp(-max(d - 100.0, 0.0) * 0.0004);
    return mix(lit, atmos_sky(view_dir, sun_dir), aerial);
}

// ---------- Waves ----------------------------------------------------------

// 7-octave sharpened-crest wave. Directions are irrationally spaced around
// the compass, frequencies are non-harmonic, time speeds and phase offsets
// are all decorrelated so there's no visible repetition pattern.

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

void wave_surface(vec2 uv, float t, out float h,
                  out float dhdx, out float dhdy) {
    // 7 octaves: one long swell + mid-frequency chops + small ripples.
    vec2 dirs[7] = vec2[](
        normalize(vec2( 1.00,  0.13)),
        normalize(vec2(-0.62,  1.00)),
        normalize(vec2( 0.78, -0.53)),
        normalize(vec2(-1.00, -0.31)),
        normalize(vec2( 0.19,  0.98)),
        normalize(vec2(-0.41, -0.91)),
        normalize(vec2( 0.88,  0.47))
    );
    // Non-harmonic frequencies (no common period between components).
    float freqs[7]  = float[]( 2.3,  5.7,  9.1, 14.9, 21.7, 31.3, 43.1);
    float amps[7]   = float[](0.120, 0.070, 0.045, 0.025, 0.014, 0.008, 0.005);
    float speeds[7] = float[](0.45, 0.83, 1.17, 1.55, 1.93, 2.41, 2.87);
    float phis[7]   = float[](0.13, 1.71, 3.24, 4.82, 0.97, 2.55, 5.33);

    h = 0.0;
    dhdx = 0.0;
    dhdy = 0.0;
    for (int i = 0; i < 7; i++) {
        vec2 d = dirs[i];
        float phase = dot(d, uv) * freqs[i] - t * speeds[i] + phis[i];
        // Sharpen crests: (cos(φ)*0.5 + 0.5)^2 peaks sharply, flat troughs.
        float c = cos(phase) * 0.5 + 0.5;
        float sh = c * c;
        h += amps[i] * (sh - 0.5);
        float g = -c * sin(phase) * amps[i] * freqs[i];
        dhdx += g * d.x;
        dhdy += g * d.y;
    }

    // Large-scale "chop mask": slowly-moving low-freq noise modulates the
    // overall amplitude so some patches of water look calm and others choppy.
    // Kills the "everything has the same waves everywhere" feel.
    float mask_n = vnoise(uv * 0.08 + vec2(t * 0.015, -t * 0.011));
    float mask = mix(0.55, 1.35, mask_n);
    h    *= mask;
    dhdx *= mask;
    dhdy *= mask;
}

// Schlick's Fresnel approximation. F0 for water ~0.02.
float fresnel_schlick(float cos_theta, float f0) {
    float x = 1.0 - clamp(cos_theta, 0.0, 1.0);
    float x2 = x * x;
    return f0 + (1.0 - f0) * x2 * x2 * x;
}

void main() {
    // Summed wave surface across two UV scales for non-grid look.
    float h1, dhdx1, dhdy1;
    float h2, dhdx2, dhdy2;
    wave_surface(v_uv_a * 8.0, time * 0.8, h1, dhdx1, dhdy1);
    wave_surface(v_uv_b * 5.0, time * 0.5, h2, dhdx2, dhdy2);
    float height = h1 + h2 * 0.6;
    vec3 N = normalize(vec3(-(dhdx1 + dhdx2 * 0.6),
                            1.5,
                            -(dhdy1 + dhdy2 * 0.6)));

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    float ndv = max(dot(N, V), 0.0);

    // Sky reflection off the wave normal. Properly Fresnel-weighted: at
    // grazing angles (small ndv) Fresnel -> 1, water acts like a mirror; looking
    // straight down (ndv -> 1) Fresnel -> 0.02, we see the blue absorption.
    vec3 refl_dir = reflect(-V, N);
    vec3 sky_refl = atmos_sky(normalize(refl_dir), sun_dir);
    float F = fresnel_schlick(ndv, 0.02);

    // Sub-surface tint (what the water "itself" looks like). Shallow blue
    // tinted by sun transmittance through a short path of water.
    vec3 shallow = vec3(0.08, 0.26, 0.38);
    vec3 deep    = vec3(0.02, 0.09, 0.18);
    // More "deep" look when the sun is low (darker water at dusk).
    float sun_k = clamp(sun_dir.y + 0.15, 0.0, 1.0);
    vec3 body = mix(deep, shallow, sun_k);
    // Ambient+diffuse on the body color (tinted by sun color).
    float ndl = max(dot(N, sun_dir), 0.0);
    body = body * (ambient_colour.rgb + light_diffuse.rgb * ndl * 0.8);

    // Specular sun glint — sharp Blinn-Phong highlight.
    vec3 H = normalize(sun_dir + V);
    float spec = pow(max(dot(N, H), 0.0), 96.0);
    vec3 glint = light_diffuse.rgb * spec * 2.5;

    // Wave-crest foam: only the rare high + steep crests. The chop_mask
    // modulated height field peaks around 0.16 in choppy patches, so we
    // threshold near the top and additionally require a steep surface
    // gradient (breaking crest) so foam stays sparse and directional.
    vec2 grad = vec2(dhdx1 + dhdx2 * 0.6, dhdy1 + dhdy2 * 0.6);
    float crest = smoothstep(0.11, 0.16, height);
    float slope = smoothstep(0.15, 0.55, length(grad));
    float foam = crest * slope * 0.55;
    vec3 foam_col = vec3(0.92, 0.94, 0.96);

    // Composite: mix body (refracted) and sky reflection by Fresnel,
    // add sun glint, then foam sits on top.
    vec3 lit = mix(body, sky_refl, F) + glint;
    lit = mix(lit, foam_col, foam);

    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 0.9);
}
