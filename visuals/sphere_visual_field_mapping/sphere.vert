attribute vec3 xyz_coordinate;
attribute float binary_state;

uniform mat4 rotation;

varying float v_state;


void main() {

    gl_Position = transform_position((rotation * vec4(xyz_coordinate, 1.0)).xyz);

    v_state = float(binary_state);
}
