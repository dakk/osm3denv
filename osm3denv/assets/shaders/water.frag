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

// Procedural ripple normal: superpose several directional sine waves,
// differentiate analytically to get a tangent-space normal.
vec3 ripple_normal(vec2 uv, float t) {
    vec2 dirs[4] = vec2[](
        normalize(vec2( 1.0,  0.3)),
        normalize(vec2(-0.6,  1.0)),
        normalize(vec2( 0.7, -0.8)),
        normalize(vec2(-1.0, -0.2))
    );
    float freqs[4] = float[](6.0, 11.0, 18.0, 24.0);
    float amps[4]  = float[](0.10, 0.06, 0.03, 0.015);
    float speeds[4] = float[](1.0, 0.7, 1.4, 1.9);

    float dhdx = 0.0;
    float dhdy = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 d = dirs[i];
        float phase = dot(d, uv) * freqs[i] - t * speeds[i];
        float k = amps[i] * freqs[i];
        float c = cos(phase) * k;
        dhdx += c * d.x;
        dhdy += c * d.y;
    }
    vec3 n = normalize(vec3(-dhdx, 1.5, -dhdy));
    return n;
}

void main() {
    vec3 n1 = ripple_normal(v_uv_a * 8.0, time * 0.8);
    vec3 n2 = ripple_normal(v_uv_b * 5.0, time * 0.5);
    vec3 N = normalize(n1 + n2);

    vec3 base = vec3(0.10, 0.28, 0.46);

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 H = normalize(sun_dir + V);
    float diffuse = max(dot(N, sun_dir), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    // Reflect sky color off the wave normal — cheap proxy for a real reflection
    // probe. Keeps water tinted by sky at dusk/dawn.
    vec3 refl_dir = reflect(-V, N);
    vec3 sky_refl = atmos_sky(normalize(refl_dir), sun_dir);

    vec3 lit = base * (ambient_colour.rgb + light_diffuse.rgb * diffuse)
             + sky_refl * 0.25
             + vec3(1.0) * spec * 0.8;
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 0.82);
}
