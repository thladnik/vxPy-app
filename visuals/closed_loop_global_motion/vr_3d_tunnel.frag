// Constants
const float PI = 3.14159265359;

// make this get the radius specified in the python file
const float tunnel_radius = 5;


// Uniform input
uniform int waveform;
uniform int motion_type;
uniform float angular_period;
uniform float fish_radial_position;
uniform float fish_axial_position;
uniform float fish_orientation;
uniform float time;

// Input
varying float v_azimuth;
varying float v_elevation;

void brightness_function(ang_pos) {

    // calculate where a ray extending from the fish oriented forward would hit the tunnel walls
    float ray_extention_angle = tan(ang_pos) * (tunnel_radius - fish_radial_position);

    // check if they it would hit a black or white part by doing floor division by half the angular period
    // and seeing if its odd or even
    floor_div = floor(ray_extention_angle * 0.5 / angular_period);

    //int c_val = int(mod(floor_div, 2.0) == 0.0 ? 1.0 : 0.0);

    float c_val;
    if (mod(floor_div, 2.0) == 0.0) {
        c_val = 1;  // Even, set color to white
    } else {
        c_val = 0;  // Odd, set color to black
    }

    return float(c_val);
}



// Main
void main() {

    // Set position to be used
    float angular_pos;
    angular_pos = v_elevation;

    // float c = 1.0 + sin(2.0 * PI * (angular_pos  / angular_period + fish_rel_position)); // I removed time as  factor to fish pos

    // Calculate brightness using position
    //float c = 1.0 + sin(2.0 * PI * (angular_pos  / angular_period + fish_rel_position)); // I removed time as  factor to fish pos
    float c = brightness_function(angular_pos);

    c /= 2.0;

    // Set final color
    c = step(c, 0.5);
    gl_FragColor = vec4(vec3(c), 1.0);

}