// Input
attribute vec3 a_position;
attribute vec2 a_texture_coord;

// Output
out vec3 v_position;
out vec2 v_texture_coord;

// Main
void main() {
    gl_Position = transform_position(a_position);
    v_position = a_position;
    v_texture_coord = a_texture_coord;
}
