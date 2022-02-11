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
    float p;
    if (motion_type == 1) {
        p = v_elevation;
    } else {
        p = v_azimuth;
    }

    // Calculate brightness using position
    float c = sin(1.0/(angular_period/360.0) * p + time * angular_velocity/angular_period *  2.0 * PI);

    // If waveform is rectangular (1): apply threshold to brightness
    if (waveform == 1) {
        c = step(c, 0.);
    }

    // Set final color
    gl_FragColor = vec4(c, c, c, 1.0);

}