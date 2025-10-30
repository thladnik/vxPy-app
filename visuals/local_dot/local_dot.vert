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
    v_position = a_position;
    gl_Position = transform_position(v_position);

    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_original_position = v_position;
    v_state = a_state;
    gl_PointSize = 10.;
}
