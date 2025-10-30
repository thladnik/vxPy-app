// Constants
const float PI = 3.14159265359;

// Uniform input
uniform float angular_period;
uniform float angular_velocity;
uniform float time;
uniform float rf_diameter;
uniform vec3 rf_center_location;

// Input
varying float v_azimuth;
varying float v_elevation;
varying vec3 v_position;



// Main
void main() {


    bool move = distance(rf_center_location, v_position) < (rf_diameter / (180.0 * 2.0)) * PI;
    //    bool move = true;

    //    float phase = v_azimuth  / angular_period;
    float phase = 180.0 * v_position.y  / angular_period;
    if(move) {
        phase += time * angular_velocity / angular_period;
    }

    // Calculate brightness using position
    float c = sin(2.0 * PI * phase);

    // Set final color
    gl_FragColor = vec4(vec3(step(c, 0.0)), 1.0);

}