#version 330 core

uniform sampler2D u_road_col;
uniform sampler2D u_road_nrm;
uniform float     u_bump_strength;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_amb_color;

#define N_SPOTS 6
uniform vec3  u_spot_pos[N_SPOTS];
uniform vec3  u_spot_color[N_SPOTS];
uniform float u_spot_cos_cutoff;
uniform vec3  u_spot_atten;  // constant, linear, quadratic

in  vec3 vWorldPos;
in  vec3 vWorldNormal;
in  vec2 vUV;
out vec4 p3d_FragColor;

void main() {
    vec3 Ng = normalize(vWorldNormal);

    vec3 albedo = texture(u_road_col, vUV).rgb;

    // Normal mapping: build TBN from geometric normal
    vec3 nrmSample = texture(u_road_nrm, vUV).rgb * 2.0 - 1.0;
    vec3 up  = abs(dot(Ng, vec3(1.0, 0.0, 0.0))) < 0.9
                   ? vec3(1.0, 0.0, 0.0)
                   : vec3(0.0, 1.0, 0.0);
    vec3 T   = normalize(up - dot(up, Ng) * Ng);
    vec3 B   = cross(Ng, T);
    mat3 TBN = mat3(T, B, Ng);
    vec3 tN  = normalize(vec3(nrmSample.xy * u_bump_strength, abs(nrmSample.z)));
    vec3 N   = normalize(TBN * tN);

    // Lambertian + ambient
    float diff  = max(dot(N, u_sun_dir), 0.0);
    // Slight specular for wet/shiny asphalt
    vec3  H     = normalize(u_sun_dir + vec3(0.0, 0.0, 1.0));
    float spec  = pow(max(dot(N, H), 0.0), 32.0) * 0.08;

    vec3 color = albedo * (u_amb_color + u_sun_color * diff * 0.85)
                 + u_sun_color * spec;

    // Spotlights (street lamps)
    vec3 spot_accum = vec3(0.0);
    for (int i = 0; i < N_SPOTS; i++) {
        if (u_spot_pos[i].z < -9000.0) continue;
        vec3  to_light  = u_spot_pos[i] - vWorldPos;
        float dist      = length(to_light);
        if (dist < 0.001) continue;
        vec3  L         = to_light / dist;
        float cos_theta = L.z;
        if (cos_theta < u_spot_cos_cutoff) continue;
        float atten = 1.0 / dot(u_spot_atten, vec3(1.0, dist, dist * dist));
        float edge  = smoothstep(u_spot_cos_cutoff, u_spot_cos_cutoff + 0.05, cos_theta);
        float sdiff = max(dot(N, L), 0.0);
        spot_accum += u_spot_color[i] * sdiff * atten * edge;
    }
    color += albedo * spot_accum;

    p3d_FragColor = vec4(color, 1.0);
}
