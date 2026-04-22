#version 330 core
// Gentle Gerstner displacement for lake/pond surfaces.
// Two small waves with short wavelengths and low amplitudes — lakes are
// calm; we rely on per-pixel normals in the fragment shader for detail.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform float osg_FrameTime;

in vec4 p3d_Vertex;

out vec3 vWorldPos;
out vec2 vOrigXY;

void gerstnerDisp(vec2 pos, float t,
                  vec2 dir, float A, float Q, float omega, float phi,
                  inout vec3 disp) {
    float theta = omega * dot(dir, pos) + phi * t;
    disp.x += Q * A * dir.x * cos(theta);
    disp.y += Q * A * dir.y * cos(theta);
    disp.z += A * sin(theta);
}

void main() {
    vec2 p = p3d_Vertex.xy;
    float t = osg_FrameTime;
    vec3 disp = vec3(0.0);

    // λ≈40 m, A=0.08 m — gentle surface undulation
    gerstnerDisp(p, t, normalize(vec2(1.0,  0.4)), 0.08, 0.20, 0.157, 1.24, disp);
    // λ≈25 m, A=0.05 m — cross ripple
    gerstnerDisp(p, t, normalize(vec2(-0.5, 1.0)), 0.05, 0.15, 0.251, 1.57, disp);

    vOrigXY   = p;
    vWorldPos = (p3d_ModelMatrix * vec4(p3d_Vertex.xyz + disp, 1.0)).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(p3d_Vertex.xyz + disp, 1.0);
}
