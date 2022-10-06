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

    mat4 rot = mat4(0.0, 1.0, 0.0, 0.0,
                    -1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0);


    vec3 pos = (rot * vec4(a_position, 1.0)).xyz;
    gl_Position = transform_position(pos);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_position = a_position;
}
