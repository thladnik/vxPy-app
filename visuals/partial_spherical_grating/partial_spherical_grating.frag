// Constants
const float PI = 3.14159265359;

// Uniform input
uniform int waveform;
uniform int motion_type;
uniform float angular_period;
uniform float angular_velocity;
uniform float time;
// Motion mask
uniform float mask_azimuth_center;
uniform float mask_azimuth_range;
uniform float mask_elevation_center;
uniform float mask_elevation_range;

// Input
varying float angular_azimuth;
varying float angular_elevation;

// Main
void main() {

    // Set position to be used
    float angular_pos;
    // Translation
    if (motion_type == 1) {
        angular_pos = angular_elevation;

    // Rotation
    } else {
        angular_pos =  angular_azimuth;
    }

    // Calculate mask
    float az_hrange = mask_azimuth_range / 2.0;
    float az_center = mask_azimuth_center;
    float el_hrange = mask_elevation_range / 2.0;
    float el_center = mask_elevation_center;

    // Check if within movement area
    bool move = (az_center - az_hrange <= angular_azimuth) && (angular_azimuth <= az_center + az_hrange);
    move = move && (el_center - el_hrange <= angular_elevation) && (angular_elevation <= el_center + el_hrange);

    // Calculate static phase
    float phase = angular_pos  / angular_period;
    // Only add to phase if within movement mask
    if(move) {
        phase -= time * angular_velocity / angular_period;
    }
    // Calcualte brightness based on phase
    float c = sin(2.0 * PI * phase);

    // If waveform is rectangular (1): apply threshold to brightness
    if (waveform == 1) {
        c = step(c, 0.);
    }

    // Set final color
    gl_FragColor = vec4(c, c, c, 1.0);

}