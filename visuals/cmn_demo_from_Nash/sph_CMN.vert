attribute vec3 a_position;   // Vertex position
attribute vec2 a_texcoord;   // texture coordinate
varying   vec2 v_texcoord;  // output

void main() {
    // Final position
    gl_Position = transform_position(a_position);

    // Assign varying variables
    v_texcoord  = a_texcoord;
}