attribute vec3 a_position;
attribute float a_texture_normal;
attribute float a_texture_dark;
attribute float a_texture_light;

uniform mat4 u_rotate;

varying vec3 v_position;
varying float v_texture_normal;
varying float v_texture_dark;
varying float v_texture_light;

void main() {

    vec4 pos = u_rotate * vec4(a_position, 1.);

    gl_Position = transform_position(pos.xyz);

    v_position = a_position;
    v_texture_normal = a_texture_normal;
    v_texture_dark = a_texture_dark;
    v_texture_light = a_texture_light;
}
