#version 120

uniform sampler2D u_texture; // The input texture
uniform float u_time; // Time uniform
uniform float u_min_value; // Minimum value in the texture
uniform float u_max_value; // Maximum value in the texture
varying vec2 v_texcoord; // Interpolated texture coordinates

void main()
{
    // Sample the texture at the interpolated texture coordinates
    vec4 color = texture2D(u_texture, v_texcoord);

    // Normalize the color based on the min and max values
    color = (color - u_min_value) / (u_max_value - u_min_value);

    // Apply some filtering or manipulation to the color here if needed

    // Output the color
    gl_FragColor = color;
}