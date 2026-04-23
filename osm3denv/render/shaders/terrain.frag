#version 330 core
// Terrain shading: texture splatting (sand / grass / rock) with per-layer
// normal mapping for surface bump detail.  Falls back gracefully to 1×1
// solid-colour textures if real assets were not downloaded.
//
// Blending logic (height + slope) is identical to the earlier procedural
// shader so the scene composition does not change — only the surface detail.

uniform float     u_origin_alt_m;   // absolute SRTM altitude of scene origin
uniform float     u_radius_m;       // half-extent of scene in metres
uniform float     u_tex_scale;      // metres per texture tile (default 20)
uniform sampler2D u_road_splatmap;   // dirt-road mask  (r=1 → road)
uniform sampler2D u_beach_splatmap;  // beach mask      (r=1 → sand)

uniform sampler2D u_sand_col;   // sand diffuse
uniform sampler2D u_sand_nrm;   // sand normal map (OpenGL, green-up)
uniform sampler2D u_grass_col;  // grass diffuse
uniform sampler2D u_grass_nrm;
uniform sampler2D u_rock_col;        // rock diffuse
uniform sampler2D u_rock_nrm;
uniform sampler2D u_beach_sand_col;  // beach-specific wet sand diffuse
uniform sampler2D u_beach_sand_nrm;
uniform float     u_bump_strength;   // XY amplifier for normal maps (default 3)

in vec3 vWorldPos;
in vec3 vWorldNormal;

out vec4 p3d_FragColor;

// -------------------------------------------------------------------------
// Noise (kept for macro colour variation + road splatmap softening)
// -------------------------------------------------------------------------

float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash(i),              hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
        f.z);
}

float fbm(vec3 p, int octaves) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < octaves; i++) {
        v += a * vnoise(p);
        p  = p * 2.13 + vec3(1.7, 9.2, 5.4);
        a *= 0.5;
    }
    return v;
}

// Procedural snow (no texture asset for snow)
vec3 snowColor(float n) {
    return vec3(0.86, 0.88, 0.92) - vec3(0.04) * n;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

void main() {
    vec3  Ng      = normalize(vWorldNormal);    // geometric normal
    float absAlt  = vWorldPos.z + u_origin_alt_m;

    float slope = dot(Ng, vec3(0.0, 0.0, 1.0)); // 1=flat, 0=vertical

    float macro = fbm(vWorldPos * 0.0018, 5);
    float micro = fbm(vWorldPos * 0.018,  4);

    // ------------------------------------------------------------------
    // Splat weights — same tuning as the procedural shader
    // ------------------------------------------------------------------
    float wSand  = 1.0 - smoothstep(-3.0 + macro * 4.0,
                                     12.0 + macro * 6.0, absAlt);
    float wSnow  = smoothstep(2000.0 - macro * 300.0,
                               2800.0 + macro * 200.0, absAlt);

    float rockThresh  = 0.68 - micro * 0.18;
    float screeThresh = rockThresh - 0.22;
    float wRock  = 1.0 - smoothstep(screeThresh + 0.05, rockThresh + 0.08, slope);
    float wScree = smoothstep(screeThresh - 0.10, screeThresh + 0.10, slope)
                 * (1.0 - smoothstep(rockThresh - 0.05, rockThresh + 0.10, slope));
    wScree *= (1.0 - wSnow);

    // Grass fills whatever sand and rock don't cover
    float wGrass = clamp(1.0 - wSand - wRock - wScree, 0.0, 1.0);

    // Normalise for texture blending (snow will override on top)
    float wTotal = wSand + wGrass + wRock + 1e-5;
    float tS = wSand  / wTotal;
    float tG = wGrass / wTotal;
    float tR = (wRock + wScree) / wTotal;

    // ------------------------------------------------------------------
    // Texture sampling — two-scale tiling + FBM UV warp
    //
    // Two scales with an irrational ratio (3.7) ensure the combined
    // repeat period (~74× the fine tile) is never visible.
    // A low-frequency FBM warp shifts the UV grid smoothly across the
    // landscape so axis-aligned repetition cannot form.
    // ------------------------------------------------------------------

    // Warp field: 2 octaves at ~250 m wavelength, magnitude ±0.2 tiles
    vec2 warp = vec2(
        fbm(vWorldPos * 0.004,                          2),
        fbm(vWorldPos * 0.004 + vec3(5.2, 1.7, 3.1),   2)
    ) - 0.5;

    vec2 uv_fine   = vWorldPos.xy / u_tex_scale         + warp * 0.40;
    vec2 uv_coarse = vWorldPos.xy / (u_tex_scale * 3.7) + warp * 0.15;

    // Helper: geometric-mean blend of two scales preserves brightness
    #define BLEND2(smp, uva, uvb) sqrt(max(texture(smp, uva).rgb * texture(smp, uvb).rgb, vec3(0.001)))
    #define BLENDNRM(smp, uva, uvb) normalize((texture(smp, uva).rgb + texture(smp, uvb).rgb) - 1.0)

    vec3 sandCol  = BLEND2(u_sand_col,  uv_fine, uv_coarse);
    vec3 grassCol = BLEND2(u_grass_col, uv_fine, uv_coarse);
    vec3 rockCol  = BLEND2(u_rock_col,  uv_fine, uv_coarse);

    // Tangent-space normals from maps (OpenGL convention: +Z out of surface)
    vec3 sandNrm  = BLENDNRM(u_sand_nrm,  uv_fine, uv_coarse);
    vec3 grassNrm = BLENDNRM(u_grass_nrm, uv_fine, uv_coarse);
    vec3 rockNrm  = BLENDNRM(u_rock_nrm,  uv_fine, uv_coarse);

    // ------------------------------------------------------------------
    // Diffuse colour — blend terrain layers, apply macro variation
    // ------------------------------------------------------------------
    vec3 color = sandCol * tS + grassCol * tG + rockCol * tR;
    color *= 0.88 + 0.24 * macro;
    color = mix(color, snowColor(micro), wSnow);

    // ------------------------------------------------------------------
    // Normal mapping — build TBN from world east/north (terrain-safe basis)
    // ------------------------------------------------------------------
    vec3 up   = abs(dot(Ng, vec3(1.0, 0.0, 0.0))) < 0.9
                    ? vec3(1.0, 0.0, 0.0)
                    : vec3(0.0, 1.0, 0.0);
    vec3 T    = normalize(up - dot(up, Ng) * Ng);
    vec3 B    = cross(Ng, T);
    mat3 TBN  = mat3(T, B, Ng);

    vec3 blended = sandNrm * tS + grassNrm * tG + rockNrm * tR;
    vec3 tNorm   = normalize(vec3(blended.xy * u_bump_strength, abs(blended.z)));
    vec3 N       = normalize(TBN * tNorm);

    // ------------------------------------------------------------------
    // Beach splatmap — directly replace the final colour and normal.
    // ------------------------------------------------------------------
    vec2  scene_uv = (vWorldPos.xy + u_radius_m) / (2.0 * u_radius_m);
    float beach_w  = texture(u_beach_splatmap, scene_uv).r;
    if (beach_w > 0.01) {
        vec3 beachCol = BLEND2(u_beach_sand_col, uv_fine, uv_coarse);
        beachCol = min(beachCol * 1.55, vec3(1.0));
        vec3 beachNrm = BLENDNRM(u_beach_sand_nrm, uv_fine, uv_coarse);
        vec3 tBeach   = normalize(vec3(beachNrm.xy * u_bump_strength, abs(beachNrm.z)));
        float w = min(beach_w * 1.5, 1.0);
        color = mix(color, beachCol, w);
        N     = normalize(mix(N, normalize(TBN * tBeach), w));
    }

    // ------------------------------------------------------------------
    // Road splatmap overlay
    // ------------------------------------------------------------------
    vec2  road_uv = scene_uv;
    float road_w  = texture(u_road_splatmap, road_uv).r;
    vec3  dirt    = vec3(0.50, 0.38, 0.22) * (0.90 + 0.20 * micro);
    color = mix(color, dirt, road_w * 0.88);

    // ------------------------------------------------------------------
    // Lighting: Lambertian diffuse + ambient (sun matches app.py light)
    // ------------------------------------------------------------------
    vec3 sunDir   = normalize(vec3(0.35, 0.60, 0.72));
    vec3 sunColor = vec3(0.95, 0.92, 0.85);
    vec3 ambColor = vec3(0.35, 0.37, 0.42);

    float diff = max(dot(N, sunDir), 0.0);
    color = color * (ambColor + sunColor * diff * 0.80);

    p3d_FragColor = vec4(color, 1.0);
}
