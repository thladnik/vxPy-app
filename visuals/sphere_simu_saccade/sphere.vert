attribute vec3 a_position;
attribute float a_azimuth;
attribute float a_elevation;
attribute float a_state;

uniform mat4 u_rotate;

varying float v_azimuth;
varying float v_elevation;
varying vec3 v_position;
varying float v_state;

void main() {

    vec4 pos = u_rotate * vec4(a_position, 1.);

    gl_Position = transform_position(pos.xyz);

    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_position = a_position;
    v_state = a_state;
}
