const float PI = 3.14159265359;

uniform float u_time;

varying vec3 v_position;
varying float v_state;
varying float v_vertex_lvl;

void main() {
//    gl_FragColor = vec4(vec3(step(v_state, .5)) * fract(v_vertex_lvl), 1.);
//    gl_FragColor = vec4(vec3(v_state), 1.);
    gl_FragColor = vec4(step(.5, vec3(v_state)), 1.);
}