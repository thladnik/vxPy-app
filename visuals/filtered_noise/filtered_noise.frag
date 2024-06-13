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

    // Output the color
    gl_FragColor = color;
}