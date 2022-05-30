const float PI = 3.14159265359;

uniform float time;
uniform float luminance;
uniform float contrast;

varying vec3 v_position;
varying float v_texture_default;

void main() {
        gl_FragColor = vec4(vec3(v_texture_default * contrast + luminance - contrast / 2.0), 1.);
}