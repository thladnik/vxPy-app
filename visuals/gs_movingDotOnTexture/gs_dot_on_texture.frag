uniform float time;
uniform float luminance;
uniform float contrast;

uniform int dot_polarity;
uniform float dot_start_angle;
uniform float dot_angular_velocity;
uniform float dot_angular_diameter;
uniform float dot_offset_angle;
uniform vec3 dot_location;  //cartesian coordinates from visual

varying float v_texture_default;
varying vec3 v_position;


#define PI 3.14159265


//vec3 sph2cart(in vec2 sph_coord)
//{
//    return vec3(sin(sph_coord.x) * cos(sph_coord.y),
//                cos(sph_coord.x) * cos(sph_coord.y),
//                sin(sph_coord.y));
//}

void main() {
//    vec3 dot_location = sph2cart(vec2((dot_start_angle + time * dot_angular_velocity) / 180.0 * PI,
//                                      dot_offset_angle / 180 * PI));
    float sec_length = dot_angular_diameter / 180 * PI; //convert dot diameter from degrees to radians
//    vec3 color = vec3(1.0 - smoothstep(sec_length, sec_length + 0.01, distance(v_position, dot_location)));
    float c = step(sec_length / 2.0, distance(v_position, dot_location));
//    vec3 color = vec3(1.0 - c);
    if(c > 0.1) {
        gl_FragColor = vec4(vec3(v_texture_default * contrast + luminance - contrast / 2.0), 1.);
        //gl_FragColor = vec4(vec3(1.0 - c), 1.0);
    } else {
        gl_FragColor = vec4(vec3(0.), 1.0);
    }
}
