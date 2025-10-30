const float PI = 3.14159265359;


varying float v_state;

void main() {
//    gl_FragColor = vec4(vec3(step(v_state, .5)) * fract(v_vertex_lvl), 1.);
//    gl_FragColor = vec4(vec3(v_state), 1.);
    gl_FragColor = vec4(step(.5, vec3(v_state)), 1.);
}