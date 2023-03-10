attribute vec3 a_position;
attribute float texture_default;

varying vec3 v_position;
varying float v_texture_default;

void main() {

    gl_Position = transform_position((a_position));

    v_position = a_position;
    v_texture_default = texture_default;
}
