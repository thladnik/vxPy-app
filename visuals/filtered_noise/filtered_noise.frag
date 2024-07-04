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
    float color = texture2D(u_texture, v_nposition).x;
    //vec4 color = texture2D(u_texture, v_nposition);

    // Output the color
    gl_FragColor = vec4(vec3(step(color, 0.5)), 1.0);
    //gl_FragColor = color;
}