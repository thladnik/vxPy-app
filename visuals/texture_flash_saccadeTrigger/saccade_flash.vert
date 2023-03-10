// Uniform Input (...)
//uniform mat4    u_model;         // Model matrix
//uniform mat4    u_view;          // View matrix
//uniform mat4    u_projection;    // Projection matrix
//uniform vec4 u_color;

// Atribute Input (...)
attribute vec3  a_position;      // Vertex position
//attribute vec4  a_color;         // Vertex color

// Varying Input (...)
//varying vec4    v_color;         // Interpolated fragment color (out)

// Do stuff
void main()
{
    //u_color = a_color;
    gl_Position = transform_position(a_position);
}