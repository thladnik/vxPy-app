const float PI = 3.14159265359;

uniform float u_time;

varying vec3 v_position;
varying float v_state;

void main() {

    gl_FragColor = vec4(vec3(v_state), 1.);
}