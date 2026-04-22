#version 330 core
// Gerstner wave vertex displacement (4 dominant waves).
// Normals are recomputed per-pixel in the fragment shader, so we only
// need to expose the displaced world-space position.
//
// Reference: GPU Gems 1, Ch.1 — "Effective Water Simulation from
//            Physical Models" (Mark Finch, Nvidia, 2004).

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform float osg_FrameTime;

in vec4 p3d_Vertex;

out vec3 vWorldPos;    // displaced, model-space == world-space (identity model)
out vec2 vOrigXY;      // undisplaced XY, used for fragment normal evaluation

// One Gerstner wave contribution to position displacement.
// dir   — unit 2-D wave direction
// A     — amplitude (metres)
// Q     — steepness ∈ [0,1]; 0 = sine, 1 = maximum
// omega — angular frequency (rad/m)
// phi   — phase speed (rad/s)
void gerstnerDisp(vec2 pos, float t,
                  vec2 dir, float A, float Q, float omega, float phi,
                  inout vec3 disp) {
    float theta = omega * dot(dir, pos) + phi * t;
    float C = cos(theta);
    float S = sin(theta);
    disp.x += Q * A * dir.x * C;
    disp.y += Q * A * dir.y * C;
    disp.z += A * S;
}

void main() {
    vec2 p = p3d_Vertex.xy;
    float t = osg_FrameTime;
    vec3 disp = vec3(0.0);

    // λ≈600 m  A=0.35 m  heading ~NE
    gerstnerDisp(p, t, normalize(vec2(1.0,  0.6)), 0.35, 0.25, 0.0105, 0.91, disp);
    // λ≈350 m  A=0.20 m  heading ~NNW
    gerstnerDisp(p, t, normalize(vec2(-0.3, 1.0)), 0.20, 0.20, 0.0180, 1.11, disp);
    // λ≈200 m  A=0.10 m  heading ~E
    gerstnerDisp(p, t, normalize(vec2(1.0, -0.25)), 0.10, 0.18, 0.0314, 1.37, disp);
    // λ≈130 m  A=0.06 m  heading ~NW
    gerstnerDisp(p, t, normalize(vec2(-0.7, 0.9)), 0.06, 0.15, 0.0483, 1.69, disp);

    vOrigXY  = p;
    vWorldPos = (p3d_ModelMatrix * vec4(p3d_Vertex.xyz + disp, 1.0)).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(p3d_Vertex.xyz + disp, 1.0);
}
