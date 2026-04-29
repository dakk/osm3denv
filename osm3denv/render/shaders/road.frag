#version 330 core

uniform sampler2D u_road_col;
uniform sampler2D u_road_nrm;
uniform float     u_bump_strength;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_amb_color;

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

    p3d_FragColor = vec4(color, 1.0);
}
