// Constants
const float PI = 3.14159265359;

// Uniform input
uniform int waveform;
uniform int motion_type;
uniform float angular_period;
uniform float angular_velocity;
uniform float time;

// Input
varying float v_azimuth;
varying float v_elevation;

// Main
void main() {

    // Set position to be used
    float angular_pos;
    if (motion_type == 1) {
        angular_pos = v_elevation;
    } else {
        angular_pos = v_azimuth;
    }

    // Calculate brightness using position
    float c = sin(2.0 * PI * (angular_pos  / angular_period + time * angular_velocity / angular_period));
//    float c = sin(1.0/(angular_period/360.0) * p + time * angular_velocity/angular_period *  2.0 * PI);

    // If waveform is rectangular (1): apply threshold to brightness
    if (waveform == 1) {
        c = step(c, 0.);
    }

    // Set final color
    gl_FragColor = vec4(c, 1.0-c, 0.0, 1.0);

}