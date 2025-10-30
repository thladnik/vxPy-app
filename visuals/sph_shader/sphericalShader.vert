// sphericalShader.vert

attribute vec3 a_position;
varying vec3 v_position;

void main() {
    // Final position
    gl_Position = transform_position(a_position);
    v_position = a_position; // a_position declared in mapped_position() include
}
