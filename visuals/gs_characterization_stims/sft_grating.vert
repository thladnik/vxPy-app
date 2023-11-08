uniform mat4 motion_axis;

// Input
attribute vec3 a_position;
attribute float a_azimuth;
attribute float a_elevation;


// Output
varying float v_azimuth;
varying float v_elevation;
varying vec3 v_position;

// Main
void main() {
    vec3 pos = (motion_axis * vec4(a_position, 1.0)).xyz;
    gl_Position = transform_position(pos);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_position = a_position;
}
