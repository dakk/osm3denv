#version 330 core

uniform float osg_FrameTime;

in vec2 v_uv;   // x = azimuth/2π, y = elevation/halfπ
in vec3 v_dir;  // normalised sphere-surface direction

out vec4 p3d_FragColor;

// ── 3-D value noise (seamless — no UV wrap seam) ─────────────────────────────

float hash31(vec3 p) {
    p  = fract(p * vec3(127.1, 311.7, 74.7));
    p += dot(p, p.yxz + 19.19);
    return fract(p.x * p.y * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i),               hash31(i + vec3(1,0,0)), u.x),
            mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), u.x), u.y),
        mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), u.x),
            mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), u.x), u.y),
        u.z);
}

float fbm(vec3 p) {
    float v = 0.0, amp = 0.5, freq = 1.0;
    for (int i = 0; i < 6; i++) {
        v    += amp * vnoise(p * freq);
        amp  *= 0.5;
        freq *= 2.1;
    }
    return v;
}

// ── main ─────────────────────────────────────────────────────────────────────

void main() {
    float elev = v_uv.y;   // 0 = horizon, 1 = zenith

    // Slow eastward cloud drift
    vec3 drift = vec3(osg_FrameTime * 0.006, osg_FrameTime * 0.002, 0.0);
    vec3 p = v_dir * 2.8 + drift;

    // Domain-warped FBM for billowy, non-repetitive cloud shapes
    vec3 warp = vec3(fbm(p + vec3(0.0, 1.3, 0.3)),
                     fbm(p + vec3(4.2, 0.0, 1.7)),
                     0.0);
    float density = fbm(p + warp * 0.5);

    // Cloud coverage (raise threshold → fewer clouds)
    float cloud = smoothstep(0.50, 0.72, density);

    // Fade clouds out toward the horizon to avoid hard dome edge
    cloud *= smoothstep(0.0, 0.20, elev);

    // Bright top, slightly blue-grey underside
    vec3 col = mix(vec3(0.72, 0.76, 0.84), vec3(1.0, 1.0, 1.0), pow(elev, 0.4));

    p3d_FragColor = vec4(col, cloud * 0.90);
}
