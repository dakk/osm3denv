#version 330 core

uniform sampler2D u_col_tex;
uniform sampler2D u_nrm_tex;
uniform float     u_bump_strength;

// Camera position derived from the view matrix inverse (column 3 = translation)
uniform mat4 p3d_ViewMatrixInverse;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_amb_color;

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vTexCoord;

out vec4 p3d_FragColor;

void main() {
    vec3 Ng  = normalize(vWorldNormal);
    vec3 col = texture(u_col_tex, vTexCoord).rgb;

    vec3 up  = abs(dot(Ng, vec3(1.0, 0.0, 0.0))) < 0.9
                   ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 T   = normalize(up - dot(up, Ng) * Ng);
    vec3 B   = cross(Ng, T);

    vec3 nt  = texture(u_nrm_tex, vTexCoord).rgb * 2.0 - 1.0;
    nt = normalize(vec3(nt.xy * u_bump_strength, abs(nt.z)));
    vec3 N   = normalize(mat3(T, B, Ng) * nt);

    float diff = max(dot(N, u_sun_dir), 0.0);
    col = col * (u_amb_color + u_sun_color * diff * 0.80);

    vec3 camPos  = p3d_ViewMatrixInverse[3].xyz;
    vec3 viewDir = normalize(camPos - vWorldPos);
    vec3 halfVec = normalize(u_sun_dir + viewDir);
    float spec   = pow(max(dot(N, halfVec), 0.0), 96.0) * 0.65;
    col += u_sun_color * spec;

    p3d_FragColor = vec4(col, 1.0);
}
