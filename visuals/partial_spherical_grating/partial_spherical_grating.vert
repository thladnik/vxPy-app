uniform mat4 motion_axis;

// Input
attribute vec3 a_position;
attribute float a_angular_azimuth;
attribute float a_angular_elevation;

// Output
varying float angular_azimuth;
varying float angular_elevation;
varying vec3 v_position;

// Main
void main() {
    vec3 pos = (motion_axis * vec4(a_position, 1.0)).xyz;
    gl_Position = transform_position(pos);
    angular_azimuth = a_angular_azimuth;
    angular_elevation = a_angular_elevation;
    v_position = a_position;
}
