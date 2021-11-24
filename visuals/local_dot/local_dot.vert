attribute vec3 a_position;
attribute float a_azimuth;
attribute float a_elevation;
attribute float a_state;

varying float v_azimuth;
varying float v_elevation;
varying vec3 v_position;
flat varying vec3 v_original_position;
varying float v_state;

void main() {

    // Final position
    gl_Position = transform_position(a_position);

    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_position = a_position;
    v_original_position = a_position;
    v_state = a_state;
    gl_PointSize = 10.;
}
