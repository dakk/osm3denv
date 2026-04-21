#version 150

// Analytical sky: single-scattering Rayleigh + Mie approximation with a
// dusk/dawn warm horizon, sun disk, a dark-blue night base, and an animated
// procedural cloud layer. View direction and sun direction are both unit
// vectors in world space; sun_dir points FROM the surface TOWARD the sun.

uniform vec4 light_direction;   // light propagation direction (from sun outward)
uniform float time;             // seconds since startup, drives cloud drift

in vec3 v_dir;
out vec4 frag_color;

// ---------- Noise helpers --------------------------------------------------

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
    for (int i = 0; i < 8; i++) {
        if (i >= octaves) break;
        sum += amp * vnoise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return sum;
}

// ---------- Base sky -------------------------------------------------------

vec3 sky_color(vec3 v, vec3 s) {
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
    float phase_m = (1.0 - g * g) /
                    pow(max(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5);
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

// ---------- Clouds ---------------------------------------------------------
//
// 2D clouds projected onto a sky plane via ``uv = v.xz / v.y``. To avoid the
// fbm-blob look we domain-warp the shape field, treat the density gradient
// as a cloud-top normal for proper sun-side-bright / shadow-side-dark
// illumination, and add a forward-scattering silver lining where the sun is
// behind a thin edge.

const float CLOUD_COVERAGE = 0.50;    // lower = fuller sky
const float CLOUD_SHARPNESS = 0.18;

// Raw density field in cloud-UV space (not the post-coverage version).
float cloud_raw(vec2 uv, float t) {
    vec2 drift = vec2(t * 0.008, t * 0.005);
    vec2 p = uv * 0.7 + drift;
    // Domain warping: a low-freq fbm bends the input of the main fbm, so
    // clouds no longer look like smooth fbm blobs.
    vec2 warp = vec2(fbm(p + vec2(1.7, 4.2), 3),
                     fbm(p + vec2(8.3, 2.8), 3)) * 2.0 - 1.0;
    float shape  = fbm(p + warp * 0.9, 5);
    float detail = fbm(p * 3.2 - drift * 0.6, 3) * 0.30;
    return shape + detail;
}

// Coverage = raw density after threshold.
float cloud_coverage(float raw) {
    return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SHARPNESS, raw);
}

// Density gradient (finite differences) → pseudo 3D normal of the cloud top.
// Scale pulls the y component up so clouds read as thick puffy shapes.
vec3 cloud_normal(vec2 uv, float t) {
    const float eps = 0.12;
    float dx = cloud_raw(uv + vec2(eps, 0.0), t)
             - cloud_raw(uv - vec2(eps, 0.0), t);
    float dy = cloud_raw(uv + vec2(0.0, eps), t)
             - cloud_raw(uv - vec2(0.0, eps), t);
    return normalize(vec3(-dx / eps, 1.6, -dy / eps));
}

// Thin high cirrus layer — very slow drift, wispy, fades to clear overhead.
float cirrus_coverage(vec2 uv, float t) {
    vec2 cuv = uv * 0.18 + vec2(t * 0.003, t * 0.002);
    float n = fbm(cuv, 4);
    return smoothstep(0.55, 0.78, n);
}

// ---------- Main -----------------------------------------------------------

void main() {
    vec3 v = normalize(v_dir);
    vec3 s = normalize(-light_direction.xyz);

    vec3 base = sky_color(v, s);

    if (v.y > 0.02) {
        vec2 uv = v.xz / max(v.y, 0.12);
        float raw = cloud_raw(uv, time);
        float cov = cloud_coverage(raw);
        cov *= smoothstep(0.02, 0.25, v.y);

        if (cov > 0.0) {
            vec3 N = cloud_normal(uv, time);

            // Diffuse lighting against the pseudo-normal so sun-facing sides
            // of each cloud look bright and shadow sides look dark. Wrap
            // slightly (+0.2) so dark sides aren't pitch black.
            float ndl = max(dot(N, s), 0.0);
            float diff = ndl * 0.9 + 0.15;

            vec3 bright = vec3(1.15, 1.10, 1.00);
            vec3 shade  = vec3(0.42, 0.47, 0.58);
            vec3 col = mix(shade, bright, smoothstep(0.1, 0.85, diff));

            // Silver lining: strong forward-scatter when the sun is near the
            // view direction, concentrated on thin edges where light passes
            // through less density.
            float sun_dot = max(dot(v, s), 0.0);
            float lining = pow(sun_dot, 14.0) * (1.0 - smoothstep(0.1, 0.7, cov));
            col += vec3(1.6, 1.30, 0.85) * lining * 1.4;

            // Dusk warmth overlay.
            float sh = s.y;
            float dusk = smoothstep(0.25, -0.05, sh);
            vec3 dusk_tint = vec3(1.25, 0.65, 0.30);
            col = mix(col, col * dusk_tint, dusk * 0.6);

            // Day/night intensity.
            float day = smoothstep(-0.10, 0.20, sh);
            col *= mix(0.12, 1.0, day);

            base = mix(base, col, cov);
        }

        // High cirrus layer, kept subtle so it reads as wisps.
        float ci = cirrus_coverage(uv, time) * smoothstep(0.1, 0.35, v.y) * 0.35;
        if (ci > 0.0) {
            vec3 cirrus_col = vec3(1.20, 1.15, 1.05);
            cirrus_col *= mix(0.10, 1.0, smoothstep(-0.10, 0.20, s.y));
            // Dusk warmth on cirrus too (they catch sun after ground is dark).
            float dusk_c = smoothstep(0.20, -0.10, s.y);
            cirrus_col = mix(cirrus_col, cirrus_col * vec3(1.35, 0.75, 0.35),
                             dusk_c * 0.7);
            base = mix(base, cirrus_col, ci);
        }
    }

    frag_color = vec4(base, 1.0);
}
