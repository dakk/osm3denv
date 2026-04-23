#version 330 core
// Procedural terrain shading: elevation + slope colour splatting with
// FBM noise variation.  No external textures required.
//
// Technique references:
//   • GPU Gems 2, Ch.2 — terrain rendering overview
//   • "Height and Slope Based Colours" — The Demon Throne dev blog
//   • Dave Hoskins hash — shadertoy.com/view/4djSRW  (MIT-like)
//   • Inigo Quilez — iquilezles.org/articles/fbm

uniform float     u_origin_alt_m;  // absolute SRTM altitude of the scene origin
uniform float     u_radius_m;      // half-extent of the scene in metres
uniform sampler2D u_road_splatmap; // dirt-road mask: 0=terrain, 1=road

in vec3 vWorldPos;
in vec3 vWorldNormal;

out vec4 p3d_FragColor;

// -------------------------------------------------------------------------
// Noise
// -------------------------------------------------------------------------

// Dave Hoskins — fast, low-correlation hash for 3-D input.
float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

// Tri-linearly interpolated value noise with Hermite smoothing.
float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);          // smoothstep curve
    return mix(
        mix(mix(hash(i),              hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
        f.z);
}

// Fractional Brownian Motion — sums octaves of value noise.
// Slight frequency offset per octave avoids axis-aligned patterns.
float fbm(vec3 p, int octaves) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < octaves; i++) {
        v += a * vnoise(p);
        p  = p * 2.13 + vec3(1.7, 9.2, 5.4);
        a *= 0.5;
    }
    return v;
}

// -------------------------------------------------------------------------
// Colour palette
// -------------------------------------------------------------------------

// Biome colour with subtle per-cell variation from n ∈ [0,1].
vec3 sandColor(float n) {
    return vec3(0.78, 0.72, 0.52) + vec3(0.06, 0.04, -0.02) * (n - 0.5);
}
vec3 grassColor(float n) {
    // Alternates between lush green and drier olive-yellow.
    return mix(vec3(0.22, 0.50, 0.14), vec3(0.40, 0.48, 0.14), n);
}
vec3 rockColor(float n) {
    return vec3(0.44, 0.40, 0.35) + vec3(0.10, 0.09, 0.08) * (n - 0.5);
}
vec3 snowColor(float n) {
    return vec3(0.86, 0.88, 0.92) - vec3(0.04) * n;
}
// Scree / loose rock on very steep faces — darker, more angular.
vec3 screeColor(float n) {
    return vec3(0.32, 0.29, 0.26) + vec3(0.08) * (n - 0.5);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

void main() {
    vec3  N       = normalize(vWorldNormal);
    float absAlt  = vWorldPos.z + u_origin_alt_m;  // metres above sea level

    // Slope: 1.0 = perfectly flat, 0.0 = vertical cliff.
    float slope = dot(N, vec3(0.0, 0.0, 1.0));

    // Large-scale noise — drives colour patch variation across the landscape.
    float macro = fbm(vWorldPos * 0.0018, 5);       // ~555 m fundamental period
    // Fine-scale noise — adds surface texture and breaks up solid fills.
    float micro = fbm(vWorldPos * 0.018,  4);       // ~55 m fundamental period

    // -----------------------------------------------------------------------
    // Elevation weights (absolute altitude in metres).
    // Transitions are wide enough that terrain with any mix of altitudes
    // looks natural without manual calibration.
    // -----------------------------------------------------------------------
    float wSand  = 1.0 - smoothstep(-3.0 + macro * 4.0,
                                     12.0 + macro * 6.0, absAlt);
    float wSnow  = smoothstep(2000.0 - macro * 300.0,
                               2800.0 + macro * 200.0, absAlt);

    // -----------------------------------------------------------------------
    // Slope weights.
    // Rock appears wherever the surface is too steep for soil to hold.
    // The threshold is modulated by noise so cliffs have ragged edges.
    // -----------------------------------------------------------------------
    float rockThresh = 0.68 - micro * 0.18;      // ~40-50° from horizontal
    float screeThresh = rockThresh - 0.22;

    float wRock  = 1.0 - smoothstep(screeThresh + 0.05, rockThresh + 0.08, slope);
    float wScree = smoothstep(screeThresh - 0.10, screeThresh + 0.10, slope)
                   * (1.0 - smoothstep(rockThresh - 0.05, rockThresh + 0.10, slope));
    wScree *= (1.0 - wSnow);   // snow covers scree at high altitude

    // -----------------------------------------------------------------------
    // Compose colour
    // -----------------------------------------------------------------------
    // Start from a noise-tinted grass base and layer upward.
    vec3 color = grassColor(macro + micro * 0.3);
    color = mix(color, screeColor(micro),  wScree);
    color = mix(color, rockColor(micro),   wRock);
    color = mix(color, sandColor(micro),   wSand);
    color = mix(color, snowColor(micro),   wSnow);

    // Subtle micro-texture brightening — breaks up large uniform patches.
    color *= 0.92 + 0.16 * micro;

    // -----------------------------------------------------------------------
    // Road splatmap — blend dirt colour where tracks/paths were rasterised.
    // UV: (0,0) = SW corner, (1,1) = NE corner, matching ENU → pixel mapping.
    // -----------------------------------------------------------------------
    vec2  road_uv = (vWorldPos.xy + u_radius_m) / (2.0 * u_radius_m);
    float road_w  = texture(u_road_splatmap, road_uv).r;
    vec3  dirt    = vec3(0.50, 0.38, 0.22) * (0.90 + 0.20 * micro);
    color = mix(color, dirt, road_w * 0.88);

    // -----------------------------------------------------------------------
    // Lighting: Lambertian diffuse + ambient.
    // Sun direction/colour match DirectionalLight HPR(-30,-50,0) in app.py.
    // -----------------------------------------------------------------------
    vec3 sunDir   = normalize(vec3(0.35, 0.60, 0.72));
    vec3 sunColor = vec3(0.95, 0.92, 0.85);
    vec3 ambColor = vec3(0.35, 0.37, 0.42);

    float diff = max(dot(N, sunDir), 0.0);
    color = color * (ambColor + sunColor * diff * 0.80);

    p3d_FragColor = vec4(color, 1.0);
}
