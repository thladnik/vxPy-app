uniform mat4 u_projection;
uniform mat4 u_model;
uniform mat4 u_view;

attribute vec3 a_position;
attribute vec4 a_rotation;

varying vec3 v_position;
varying vec4 v_rotation;

void main() {
    vec4 position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    gl_Position = position;
    v_position = a_position;
    v_rotation = a_rotation;
}