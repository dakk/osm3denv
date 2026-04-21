#version 150

// Analytical sky: single-scattering Rayleigh + Mie approximation, with a
// dusk/dawn warm horizon, sun disk, and a dark-blue night base.
// View direction and sun direction are both unit vectors in world space;
// sun_dir points FROM the surface TOWARD the sun.

uniform vec4 light_direction;   // light propagation direction (from sun outward)

in vec3 v_dir;
out vec4 frag_color;

vec3 sky_color(vec3 v, vec3 s) {
    float h = v.y;
    float sh = s.y;
    float mu = clamp(dot(v, s), -1.0, 1.0);

    // Base day gradient: lighter at horizon, deeper blue overhead.
    vec3 zenith  = vec3(0.22, 0.48, 0.85);
    vec3 horizon = vec3(0.75, 0.85, 0.92);
    vec3 day_col = mix(horizon, zenith, smoothstep(0.0, 0.6, max(h, 0.0)));

    // Soft warm halo centred on the sun.
    float ang = acos(mu);
    float halo = exp(-ang * 6.0);
    vec3 halo_tint = vec3(1.00, 0.88, 0.65);
    day_col = mix(day_col, day_col + halo_tint * 0.5, halo * 0.6);

    // Tight Mie forward peak for extra brightness right at the sun.
    float g = 0.82;
    float phase_m = (1.0 - g * g) /
                    pow(max(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5);
    day_col += halo_tint * phase_m * 0.005;

    // Sun disk (~0.5°).
    float sun_disk = smoothstep(0.9996, 0.9999, mu);
    day_col = mix(day_col, vec3(1.40, 1.25, 1.00), sun_disk);

    // Warm horizon at dawn/dusk.
    float dusk = smoothstep(0.25, -0.05, sh);
    float near_h = 1.0 - smoothstep(0.0, 0.35, max(h, 0.0));
    vec3 dusk_tint = vec3(1.10, 0.55, 0.25);
    day_col = mix(day_col, dusk_tint, dusk * near_h * 0.75);

    // Day/night blend keyed to sun elevation.
    float day = smoothstep(-0.10, 0.20, sh);
    vec3 night = vec3(0.02, 0.03, 0.06);
    vec3 C = mix(night, day_col, day);

    // Fade the dome below the horizon plane.
    C *= smoothstep(-0.20, 0.0, h) * 0.55 + 0.45;
    return C;
}

void main() {
    vec3 v = normalize(v_dir);
    vec3 s = normalize(-light_direction.xyz);
    frag_color = vec4(sky_color(v, s), 1.0);
}
