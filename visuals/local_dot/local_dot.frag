const float PI = 3.14159265359;

uniform float u_time;

varying vec3 v_position;
flat varying vec3 v_original_position;
varying float v_state;

float disc(vec2 P, float size   )
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= 8.0/2.0;
    return r;
}

void main() {

//    gl_FragColor = vec4(vec3(v_state), 1.);

//    float size = 10.;
//    float r = disc(gl_PointCoord, size);
//    float d = abs(r);
//    if(r > 2.0 || v_state == 0.0)
//        discard;
//    else
//        gl_FragColor = vec4(vec3(v_state), 1.);

    float dot_diameter = 2.;
//    float d = distance(v_position, v_original_position);
    float angle = acos(dot(normalize(v_position), normalize(v_original_position)));
    float c = step(angle, dot_diameter / 180. * PI / 2.);

    if (c == 0.0)
        discard;
    else
        gl_FragColor = vec4(vec3(c), 1.);

}