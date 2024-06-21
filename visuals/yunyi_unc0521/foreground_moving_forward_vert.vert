// Input
attribute vec3 a_position;
attribute vec2 a_texcoord;   // texture coordinate

attribute float a_azimuth;
attribute float a_elevation;
attribute vec2 a_texcoord_fore;
attribute vec2 a_texture_start_coords;
attribute vec2 a_sphere_texture_start_coords;
attribute float a_tile_center_x;
attribute float a_tile_center_y;


//uniform float x
//uniform float y
varying   vec2 v_texcoord;  // output
varying vec2 v_texcoord_fore;
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 v_texture_start_coords;
varying vec2 v_sphere_texture_start_coords;
varying vec2 vertices1;
varying vec2 vertices2;
varying vec2 vertices3;
#define PI 3.14159265


// Output
varying float v_azimuth;
varying float v_elevation;

// Main
void main() {
    float triangle_size = 0.1;
    gl_Position = transform_position(a_position);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_texcoord  = a_texcoord;
    v_texcoord_fore = vec2(a_azimuth, a_elevation);
    v_texture_start_coords = a_texture_start_coords;
    v_sphere_texture_start_coords = a_sphere_texture_start_coords;
    v_tile_center_x = a_tile_center_x;
    v_tile_center_y = a_tile_center_y;
    vertices1 = vec2(a_tile_center_x - triangle_size / 2 * sqrt(3), a_tile_center_y - triangle_size / 2);
    vertices2 = vec2(a_tile_center_x + triangle_size / 2 * sqrt(3), a_tile_center_y - triangle_size / 2);
    vertices3 = vec2(a_tile_center_x, a_tile_center_y + triangle_size / 2 * sqrt(3));
}


