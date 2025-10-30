uniform mat4 motion_axis;
uniform mat4 rotation;

// Input
attribute vec3 a_position;
attribute float texture_default;


// Output
varying vec3 v_position;
varying float v_texture_default;

// Main
void main() {
    vec3 pos = (motion_axis * vec4(a_position, 1.0)).xyz;
    gl_Position = transform_position((rotation * vec4(a_position, 1.0)).xyz);

    v_texture_default = texture_default;
    v_position = a_position;
}
