// Input
attribute vec3 a_position;

// Output
varying vec3 v_position;

// Main
void main() {
    gl_Position = transform_position(a_position);
    v_position = a_position;
}
