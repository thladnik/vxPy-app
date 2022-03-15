// Constants
const float PI = 3.14159265359;

// Uniform input
uniform int waveform;
uniform int direction;
uniform float linear_velocity;
uniform float spatial_period;
uniform float time;

// Input
varying vec2 v_position;

void main() {

    // Set position to be used
    float p;
    if (direction == 1) {
        p = v_position.y;
    } else {
        p = v_position.x;
    }

    // Calculate brightness using position
    float c = sin((p + time * linear_velocity)/spatial_period * 2.0 * PI);

    // If shape is rectangular (1): apply threshold to brightness
    if(waveform == 1) {
        c = step(c, 0.);
    }

    // Set final color
    gl_FragColor = vec4(c, c, c, 1.0);
}
