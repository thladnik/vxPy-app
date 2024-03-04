uniform float time;

uniform int dot_polarity;
uniform float dot_start_angle;
uniform float dot_angular_diameter;
uniform vec3 dot_location;

varying vec3 v_position;


#define PI 3.14159265


void main() {
    float sec_length = sin(dot_angular_diameter / 360 * PI);
    float c = smoothstep(sec_length, sec_length + 0.01, distance(v_position, dot_location));

    if(dot_polarity == 1) {
        gl_FragColor = vec4(vec3(c), 1.0);
    } else {
        gl_FragColor = vec4(vec3(1.0 - c), 1.0);
    }
}
