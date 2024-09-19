// Input
attribute vec3 a_position;
attribute vec2 a_texcoord;   // texture coordinate
attribute vec3 a_tile_center;
attribute float a_azimuth;
attribute float a_elevation;
attribute vec2 a_texcoord_fore;
attribute float a_tile_center_x;
attribute float a_tile_center_y;
attribute vec2 a_texture_start_coords;

//uniform float x
//uniform float y
varying   vec2 v_texcoord;  // output
varying vec2 v_texcoord_fore;
varying vec3 v_tile_center;
#define PI 3.14159265
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 v_texture_start_coords;


// Output
varying float v_azimuth;
varying float v_elevation;

// Main
void main() {
    gl_Position = transform_position(a_position);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_texcoord  = a_texcoord;
    v_texcoord_fore = vec2(a_azimuth, a_elevation);
    v_tile_center_x = a_tile_center_x;
    v_tile_center_y = a_tile_center_y;
    v_texture_start_coords = a_texture_start_coords;
    v_tile_center = a_tile_center;
}


