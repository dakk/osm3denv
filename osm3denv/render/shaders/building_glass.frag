#version 330 core

uniform sampler2D u_col_tex;
uniform sampler2D u_nrm_tex;
uniform float     u_bump_strength;

// Camera position derived from the view matrix inverse (column 3 = translation)
uniform mat4 p3d_ViewMatrixInverse;

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

    vec3 sunDir   = normalize(vec3(0.35, 0.60, 0.72));
    vec3 sunColor = vec3(0.95, 0.92, 0.85);
    vec3 ambColor = vec3(0.35, 0.37, 0.42);

    float diff = max(dot(N, sunDir), 0.0);
    col = col * (ambColor + sunColor * diff * 0.80);

    // Blinn-Phong specular using camera position from the view matrix inverse
    vec3 camPos  = p3d_ViewMatrixInverse[3].xyz;
    vec3 viewDir = normalize(camPos - vWorldPos);
    vec3 halfVec = normalize(sunDir + viewDir);
    float spec   = pow(max(dot(N, halfVec), 0.0), 96.0) * 0.65;
    col += sunColor * spec;

    p3d_FragColor = vec4(col, 1.0);
}
