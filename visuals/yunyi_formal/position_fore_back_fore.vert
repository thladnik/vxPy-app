// Input
attribute vec3 a_position;
attribute vec2 a_texcoord;   // texture coordinate
attribute vec2 a_texcoord_fore;
attribute float a_azimuth;
attribute float a_elevation;
varying   vec2 v_texcoord;  // output
varying vec2  v_texcoord_fore;


// Output
varying float v_azimuth;
varying float v_elevation;

// Main
void main() {
    gl_Position = transform_position(a_position);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_texcoord  = a_texcoord;
    v_texcoord_fore = a_texcoord_fore;
}


