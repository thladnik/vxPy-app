const float PI = 3.14159265359;

uniform float u_time;
uniform int u_flash_polarity;

varying vec3 v_position;
varying float v_texture_normal;
varying float v_texture_dark;
varying float v_texture_light;

void main() {
    if(u_flash_polarity == 0) {
        gl_FragColor = vec4(vec3(v_texture_normal), 1.);
    } else if(u_flash_polarity == 1) {
        gl_FragColor = vec4(vec3(v_texture_light), 1.);
    } else { // Should be -1
        gl_FragColor = vec4(vec3(v_texture_dark), 1.);
    }
}