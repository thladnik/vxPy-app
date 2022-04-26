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
    angular_pos = v_elevation;

    // Calculate brightness using position
    float c = 1.0 + sin(2.0 * PI * (angular_pos  / angular_period + time * angular_velocity / angular_period));
    c /= 2.0;

    // Set final color
    c = step(c, 0.5);
    gl_FragColor = vec4(vec3(c), 1.0);

}