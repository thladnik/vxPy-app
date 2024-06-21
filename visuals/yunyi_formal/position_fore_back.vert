// Input
attribute vec3 a_position;
attribute vec2 a_texcoord;   // texture coordinate
attribute float a_azimuth;
attribute float a_elevation;
varying   vec2 v_texcoord;  // output


// Output
varying float v_azimuth;
varying float v_elevation;

// Main
void main() {
    gl_Position = transform_position(a_position);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_texcoord  = a_texcoord;
}


