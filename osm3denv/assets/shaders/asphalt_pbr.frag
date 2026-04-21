#version 150

// Textured PBR asphalt. Samples an ambientCG pack (Color / NormalGL /
// Roughness / AmbientOcclusion) and feeds the values through the same
// analytical Cook-Torrance BRDF + sky-IBL approximation we use elsewhere.
// Tangent basis is derived per-pixel from world-position derivatives so the
// mesh can skip per-vertex tangents.

uniform vec4 ambient_colour;
uniform vec4 light_diffuse;
uniform vec4 light_direction;
uniform vec3 camera_position;

uniform sampler2D albedo_map;
uniform sampler2D normal_map;
uniform sampler2D rough_map;
uniform sampler2D ao_map;

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
    float aerial = 1.0 - exp(-max(d - 100.0, 0.0) * 0.0004);
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
    vec3  F = pbr_F_schlick(vdh, F0);

    vec3 spec = (D * G * F) / max(4.0 * ndv * ndl, 1e-4);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kd * albedo / PBR_PI;
    vec3 direct = (diff + spec) * ndl * sun_color;

    vec3 R = reflect(-V, N);
    vec3 env_diff = atmos_sky(N, L) * albedo * (1.0 - metallic) * 0.35;
    vec3 env_spec = atmos_sky(normalize(R), L)
                  * pbr_F_schlick(ndv, F0)
                  * (1.0 - roughness * 0.9) * 0.35;
    vec3 floor_amb = albedo * ambient_col * (1.0 - metallic) * 0.5;

    return direct + env_diff + env_spec + floor_amb;
}

// ---------- Tangent basis from derivatives ---------------------------------
//
// Per-pixel cotangent frame (Christian Schüler). Lets us apply tangent-space
// normal maps without shipping tangents in the vertex data.

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

void main() {
    // Tile size in world metres: ambientCG Asphalt026A represents a 2 m patch,
    // so divide world-UV by 2 to get one full texture per 2 m of road.
    vec2 uv = v_uv * 0.5;

    vec3 albedo = texture(albedo_map, uv).rgb;
    float rough = clamp(texture(rough_map, uv).r, 0.04, 1.0);
    float ao    = texture(ao_map, uv).r;

    // Break up visible tiling with a subtle low-frequency luminance warp.
    // No noise helpers in scope here — sample one more octave of the albedo
    // texture at a very different scale and blend it in.
    vec3 macro = texture(albedo_map, uv * 0.17).rgb;
    float macro_lum = dot(macro, vec3(0.299, 0.587, 0.114));
    albedo *= mix(0.85, 1.15, macro_lum);

    vec3 N_geo = normalize(v_world_normal);
    vec3 n_tangent = texture(normal_map, uv).rgb * 2.0 - 1.0;
    // Normal-map strength (reduce if too bumpy).
    n_tangent.xy *= 1.0;
    mat3 TBN = cotangent_frame(N_geo, v_world_pos, uv);
    vec3 N = normalize(TBN * n_tangent);

    vec3 sun_dir = normalize(-light_direction.xyz);
    vec3 V = normalize(camera_position - v_world_pos);
    vec3 lit = pbr_surface(albedo * ao, N, V, sun_dir,
                           light_diffuse.rgb, ambient_colour.rgb,
                           rough, 0.0);
    vec3 final = apply_aerial(lit, v_world_pos, camera_position, sun_dir);
    frag_color = vec4(final, 1.0);
}
