// Constants
const float PI = 3.14159265359;

// Uniform input
uniform float grating_angular_period;
uniform float grating_angular_velocity;
uniform float time;
uniform float stimulus_patch_diameter;
uniform vec3 stimulus_patch_center;
uniform mat4 stimulus_patch_rotation;
uniform mat4 grating_direction_rotation;

// Input
varying vec3 v_position;

vec3 calculate_spherical_coords(vec3 pos) {

    float theta = atan(pos.y, pos.x);
    float phi = atan(sqrt(pos.x * pos.x + pos.y * pos.y), pos.z);
    float r = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

    return vec3(theta, phi, r);

}

// Main
void main() {

    // Rotate the vector pointing in the direction of the patch center
    vec3 rot_patch_center = (stimulus_patch_rotation * vec4(stimulus_patch_center, 1.0)).xyz;

    // Determine if position lies within patch (as defined by patch center and diameter)
    bool move = distance(rot_patch_center, v_position) < (stimulus_patch_diameter / (180.0 * 2.0)) * PI;

    // Calculate rotated position for texture
    vec3 rot_pos = (grating_direction_rotation * vec4(v_position, 1.0)).xyz;

    // Calculate phase at position
    float phase = calculate_spherical_coords(rot_pos).x / PI * 180.0 / grating_angular_period;

    // Advance phase based on elapsed time and stimulus velocity
    phase += float(move) * time * grating_angular_velocity / grating_angular_period;

    // Calculate brightness using position
    float c = sin(2.0 * PI * phase);

    // Set final binary color
    gl_FragColor = vec4(vec3(step(c, 0.0)), 1.0);

}