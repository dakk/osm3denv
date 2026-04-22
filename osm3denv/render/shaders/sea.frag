#version 330 core
// Per-pixel Gerstner normal from 8 wave layers → Schlick Fresnel + Blinn-Phong.
// Normals are evaluated analytically from the same wave formula used in the
// vertex shader (4 dominant waves) plus 4 extra high-frequency waves that
// are too fine to tessellate but contribute visually to specular highlights.
//
// Reference: GPU Gems 1, Ch.1 — "Effective Water Simulation from
//            Physical Models" (Mark Finch, Nvidia, 2004).

uniform float osg_FrameTime;
uniform mat4 p3d_ViewMatrixInverse;

in vec3 vWorldPos;
in vec2 vOrigXY;

out vec4 p3d_FragColor;

// Accumulate the normal derivative for one Gerstner wave.
// Uses the undisplaced XY as the phase argument — error is negligible
// when displacement << wavelength.
void gerstnerNormal(vec2 pos, float t,
                    vec2 dir, float A, float Q, float omega, float phi,
                    inout vec3 N) {
    float theta = omega * dot(dir, pos) + phi * t;
    float C = cos(theta);
    float S = sin(theta);
    N.x -= dir.x * omega * A * C;
    N.y -= dir.y * omega * A * C;
    N.z -= Q * omega * A * S;
}

vec3 waterNormal(vec2 p, float t) {
    vec3 N = vec3(0.0);

    // ---- 4 dominant waves (match vertex shader) ----
    gerstnerNormal(p, t, normalize(vec2( 1.0,  0.6)),  1.00, 0.35, 0.0105, 0.91, N);
    gerstnerNormal(p, t, normalize(vec2(-0.3,  1.0)),  0.60, 0.30, 0.0180, 1.11, N);
    gerstnerNormal(p, t, normalize(vec2( 1.0, -0.25)), 0.35, 0.25, 0.0314, 1.37, N);
    gerstnerNormal(p, t, normalize(vec2(-0.7,  0.9)),  0.20, 0.22, 0.0483, 1.69, N);

    // ---- 4 high-frequency detail waves (fragment only) ----
    // λ≈70 m  — fine chop
    gerstnerNormal(p, t, normalize(vec2( 0.8, -0.6)),  0.12, 0.20, 0.0898, 2.58, N);
    // λ≈40 m
    gerstnerNormal(p, t, normalize(vec2(-0.5, -1.0)),  0.08, 0.18, 0.1571, 3.41, N);
    // λ≈20 m
    gerstnerNormal(p, t, normalize(vec2( 0.4,  1.0)),  0.05, 0.15, 0.3142, 4.82, N);
    // λ≈12 m  — surface ripple
    gerstnerNormal(p, t, normalize(vec2(-1.0,  0.3)),  0.03, 0.12, 0.5236, 6.23, N);

    return normalize(vec3(-N.x, -N.y, 1.0 - N.z));
}

void main() {
    float t   = osg_FrameTime;
    vec3  N   = waterNormal(vOrigXY, t);
    vec3  cam = p3d_ViewMatrixInverse[3].xyz;
    vec3  V   = normalize(cam - vWorldPos);

    // Sun direction (toward sun) — matches DirectionalLight HPR(-30,-50,0).
    vec3 sunDir   = normalize(vec3(0.35, 0.60, 0.72));
    vec3 sunColor = vec3(1.00, 0.96, 0.88);

    // Sky colour (approximates background, so grazing Fresnel blends into sky).
    vec3 skyColor = vec3(0.53, 0.70, 0.86);

    // Schlick Fresnel — F0=0.04 (water/air interface).
    float cosA   = max(dot(N, V), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - cosA, 5.0);

    // Blinn-Phong specular.
    vec3  H    = normalize(sunDir + V);
    float spec = pow(max(dot(N, H), 0.0), 160.0);

    // Wave steepness scalar — drives "whitecap" tinting near crests.
    // N.z ≈ 1 on a flat surface, decreases as waves steepen.
    float steepness = 1.0 - N.z;
    vec3 foam = vec3(0.95, 0.97, 1.0) * smoothstep(0.35, 0.65, steepness) * 0.4;

    // Base water colour: deep blue at normal incidence, teal at grazing.
    vec3 deepColor    = vec3(0.03, 0.12, 0.24);
    vec3 shallowColor = vec3(0.07, 0.38, 0.52);
    vec3 waterColor   = mix(deepColor, shallowColor, fresnel * 0.5 + 0.12);

    // Fresnel reflection blends water body toward sky colour.
    vec3 color = mix(waterColor, skyColor, fresnel * 0.55);
    color += spec * sunColor * 0.90;
    color += foam;

    p3d_FragColor = vec4(color, 1.0);
}
