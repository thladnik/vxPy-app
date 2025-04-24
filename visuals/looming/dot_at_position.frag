uniform float time;

uniform int dot_polarity;
uniform float dot_angular_diameter;
uniform float dot_azimuth;
uniform float dot_elevation;

varying vec3 v_position;


#define PI 3.14159265

vec3 sph2cart(vec2 sph_coords) {

    float az = sph_coords.x / 180.0 * PI;
    float el = sph_coords.y / 180.0 * PI;

    float x = cos(az) * cos(el);
    float y = sin(az) * cos(el);
    float z = sin(el);

    return vec3(x, y, z);
}


void main() {
    float sec_length = dot_angular_diameter / 360 * PI;
    float c = step(sec_length / 2.0, distance(v_position, sph2cart(vec2(dot_azimuth, dot_elevation))));

    if(dot_polarity == 1) {
        gl_FragColor = vec4(vec3(c), 1.0);
    } else {
        gl_FragColor = vec4(vec3(1.0 - c), 1.0);
    }
}
