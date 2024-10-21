#define PI 3.1415926535897932384626433832795

uniform mat4 rotation_matrix;
uniform float spatial_period;
uniform vec3 rotation_axis;

varying vec3 v_position;

void main() {

    vec3 position = (rotation_matrix * vec4(v_position, 1.0)).xyz;

    float azim = atan(position.z, position.y);

    float c = sin(2.0 * PI * azim / radians(spatial_period));

    gl_FragColor = vec4(vec3(c), 1.0);

}