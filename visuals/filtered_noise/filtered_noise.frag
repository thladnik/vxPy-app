// Input
varying vec2 v_position;
varying vec2 v_nposition;

// Uniforms
uniform sampler2D u_texture;
uniform float u_min_value;
uniform float u_max_value;

// Main
void main() {
    // Sample the texture
    vec4 color = texture2D(u_texture, v_nposition);

    // Normalize the color value
    color = (color - u_min_value) / (u_max_value - u_min_value);

    // Output the color
    gl_FragColor = color;
}