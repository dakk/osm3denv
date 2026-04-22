#version 330 core
// Per-pixel Gerstner normals for a calm lake/pond surface.
// Shorter wavelengths and lower amplitudes than the sea shader.
// Freshwater colour palette: dark teal-green depths, lighter blue-green
// at grazing angles where the sky is reflected.

uniform float osg_FrameTime;
uniform mat4 p3d_ViewMatrixInverse;

in vec3 vWorldPos;
in vec2 vOrigXY;

out vec4 p3d_FragColor;

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

vec3 lakeNormal(vec2 p, float t) {
    vec3 N = vec3(0.0);

    // Match the two vertex-shader waves.
    gerstnerNormal(p, t, normalize(vec2( 1.0,  0.4)), 0.08, 0.20, 0.157, 1.24, N);
    gerstnerNormal(p, t, normalize(vec2(-0.5,  1.0)), 0.05, 0.15, 0.251, 1.57, N);

    // Extra high-frequency ripples (fragment only).
    // λ≈12 m
    gerstnerNormal(p, t, normalize(vec2( 0.7, -0.7)), 0.03, 0.12, 0.524, 2.21, N);
    // λ≈7 m
    gerstnerNormal(p, t, normalize(vec2(-0.9,  0.4)), 0.02, 0.10, 0.898, 2.89, N);
    // λ≈4 m — finest surface texture
    gerstnerNormal(p, t, normalize(vec2( 0.3,  1.0)), 0.01, 0.08, 1.571, 3.83, N);

    return normalize(vec3(-N.x, -N.y, 1.0 - N.z));
}

void main() {
    float t = osg_FrameTime;
    vec3  N = lakeNormal(vOrigXY, t);
    vec3  cam = p3d_ViewMatrixInverse[3].xyz;
    vec3  V   = normalize(cam - vWorldPos);

    // Sun direction — matches DirectionalLight HPR(-30,-50,0).
    vec3 sunDir   = normalize(vec3(0.35, 0.60, 0.72));
    vec3 sunColor = vec3(1.00, 0.96, 0.88);

    // Sky colour for Fresnel reflection approximation.
    vec3 skyColor = vec3(0.53, 0.70, 0.86);

    // Schlick Fresnel, F0=0.04.
    float cosA    = max(dot(N, V), 0.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - cosA, 5.0);

    // Blinn-Phong specular — softer than sea (lower exponent, lower intensity).
    vec3  H    = normalize(sunDir + V);
    float spec = pow(max(dot(N, H), 0.0), 80.0);

    // Freshwater colours: dark green-teal deep, lighter blue-green at edges.
    vec3 deepColor    = vec3(0.03, 0.14, 0.17);
    vec3 shallowColor = vec3(0.12, 0.42, 0.38);
    vec3 waterColor   = mix(deepColor, shallowColor, fresnel * 0.45 + 0.10);

    // Fresnel reflection toward sky.
    vec3 color = mix(waterColor, skyColor, fresnel * 0.50);
    color += spec * sunColor * 0.55;

    p3d_FragColor = vec4(color, 1.0);
}
