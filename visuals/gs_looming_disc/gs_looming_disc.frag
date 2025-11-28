uniform float time;
uniform float luminance;
uniform float contrast;

uniform int disc_polarity;
uniform float disc_diameter;    //in degrees
uniform float disc_current_azimuth; //in degrees
uniform float disc_elevation;   //in degrees


varying float v_texture_default;
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
    float sec_length = disc_diameter / 180 * PI;    //convert disc diameter from degrees to radians
    float c = step(sec_length / 2.0, distance(v_position, sph2cart(vec2(disc_current_azimuth, disc_elevation))));
    //float c = smoothstep(sec_length, sec_length + 0.01, distance(v_position, sph2cart(vec2(disc_current_azimuth, disc_elevation))));


    if(c > 0.1) {
        gl_FragColor = vec4(vec3(v_texture_default * contrast + luminance - contrast / 2.0), 1.);
    } else {
        gl_FragColor = vec4(vec3(0.), 1.0);
    }
}
