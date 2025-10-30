const float PI = 3.14159265359;

uniform float u_time;
uniform int u_flash_polarity;
uniform float time;
uniform float luminance;
uniform float contrast;

varying vec3 v_position;
varying float v_texture_normal;
varying float v_texture_dark;
varying float v_texture_light;

void main() {
        gl_FragColor = vec4(vec3(v_texture_normal * contrast + luminance - contrast / 2.0), 1.);
}