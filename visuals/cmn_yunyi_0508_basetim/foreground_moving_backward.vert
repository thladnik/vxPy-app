uniform mat4 u_projection;
uniform mat4 u_model;
uniform mat4 u_view;
attribute vec4 a_start_rotation;
attribute vec3 a_position;
attribute vec4 a_rotation;
attribute vec4 a_rotation_stationary;
attribute float a_triangle_center_az;
attribute float a_triangle_center_el;
varying vec3 v_position;
varying vec4 v_rotation;
varying float v_triangle_center_az;
varying float v_triangle_center_el;
varying vec4 v_start_rotation;
varying vec4 v_rotation_stationary;

void main() {
    vec4 position = transform_position(a_position);
    gl_Position = position;
    v_position = 2*a_position;
    v_rotation = a_rotation;
    v_triangle_center_az = a_triangle_center_az;
    v_triangle_center_el = a_triangle_center_el;
    v_start_rotation = a_start_rotation;
    v_rotation_stationary = a_rotation_stationary;
}