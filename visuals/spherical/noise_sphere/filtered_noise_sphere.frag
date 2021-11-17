const float PI = 3.14159265359;

uniform float u_time;

varying vec3 v_position;
varying float v_state;

void main() {

    // Angle between dot center location and v_position
//    float angle = acos(dot(normalize(v_position), normalize(u_dot_location)));

    // Threshold angle with u_dot_diameter
//    float c = step(angle, u_dot_diameter / 180. * PI / 2.);

//    if(c > .01) {
//        gl_FragColor = vec4(vec3(1.), 1.);
//    } else {
//        discard;
//    }

//    vec3 pos = gl_FragCoord.xyz;

    vec3 vx = dFdx(v_position);
    float dx = length(vx-v_position);
    vec3 vy = dFdy(v_position);
    float dy = length(vy-v_position);
    float d = (dx + dy) / 2.;

    float ds = (dFdx(v_state) + dFdy(v_state)) / 2.;
    float sign = step(ds, 0.) * 2. - 1.;

    gl_FragColor = vec4(vec3(v_state * cos(length(dx)) * cos(length(dy))), 1.);
//    gl_FragColor = vec4(vec3(1. / distance(pos, v_position)), 1.0);
}