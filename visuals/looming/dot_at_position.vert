uniform mat4 static_rotation;

// Input
attribute vec3 a_position;


// Output
varying vec3 v_position;

// Main
void main() {
    vec4 pos = static_rotation * vec4(a_position, 1.0);
    gl_Position = transform_position(pos.xyz);

    v_position = a_position;
}
