#version 150

// Sky dome. Mesh is an inverted unit cube attached to a scene node that
// inherits only the camera's *position* (not orientation), so each vertex's
// object-local position equals the world-space view direction.

uniform mat4 world_view_proj_matrix;

in vec4 position;

out vec3 v_dir;

void main() {
    vec4 clip = world_view_proj_matrix * position;
    // Force z = w so the dome writes at the far plane; anything else in the
    // scene depth-tests in front of it.
    gl_Position = clip.xyww;
    v_dir = position.xyz;
}
