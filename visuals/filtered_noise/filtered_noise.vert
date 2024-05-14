attribute vec3 a_position; // Changed from vec2 to vec3
varying vec2 v_texcoord;

void main()
{
    // Pass the vertex position to the fragment shader
    v_texcoord = a_position.xy * 0.5 + 0.5; // Adjusted to extract only the x and y components
    // Set the vertex position in clip space
    gl_Position = vec4(a_position, 1.0); // Changed 0.0 to 1.0 for the w component
}