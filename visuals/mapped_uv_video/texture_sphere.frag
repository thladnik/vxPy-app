#version 460

uniform sampler2D video_texture;

// Input
varying vec3 v_position;
varying vec2 v_texture_coord;

// Main
void main() {
    gl_FragColor = texture(video_texture, v_texture_coord);
}
