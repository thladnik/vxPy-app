// Input
attribute vec3 a_position;
attribute vec2 a_texcoord;   // texture coordinate
attribute float a_azimuth;
attribute float a_elevation;
attribute float a_tile_center_x;
attribute float a_tile_center_y;
attribute vec2 a_texcoord_fore;
attribute vec2 a_texture_start_coords;
attribute float a_triangle_num;
attribute vec2 a_texture_foreground_cordinate2D;

//uniform float x
//uniform float y
varying vec2 v_texcoord;  // output
varying vec2 v_texcoord_fore;
varying vec2 v_tile_center;
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 vertices1;
varying vec2 vertices2;
varying vec2 vertices3;
varying vec2 v_texture_start_coords;
varying float v_triangle_num;
varying float triangle_size;
#define PI 3.14159265


// Output
varying float v_azimuth;
varying float v_elevation;
varying vec2 v_texture_foreground_cordinate2D;

// Main
void main() {
    float triangle_size = 0.1;
    gl_Position = transform_position(a_position);
    v_azimuth = a_azimuth;
    v_elevation = a_elevation;
    v_texcoord  = a_texcoord;
    v_texcoord_fore = vec2(a_azimuth, a_elevation);
    v_tile_center = vec2(a_tile_center_x, a_tile_center_y) ;
    v_tile_center_x = a_tile_center_x;
    v_tile_center_y = a_tile_center_y;
    v_texture_start_coords = a_texture_start_coords;
    v_triangle_num = a_triangle_num;
    vertices1 = vec2(a_tile_center_x - triangle_size / 2 * sqrt(3), a_tile_center_y - triangle_size / 2);
    vertices2 = vec2(a_tile_center_x + triangle_size / 2 * sqrt(3), a_tile_center_y - triangle_size / 2);
    vertices3 = vec2(a_tile_center_x, a_tile_center_y + triangle_size / 2 * sqrt(3));
    //texture_new_cords = mat3(vertices1, vertices2, vertices3);
    v_texture_foreground_cordinate2D = a_texture_foreground_cordinate2D;
}


