#version 410

uniform sampler2D video_texture;

// Input
in vec3 v_position;
in vec2 v_texture_coord;

// Main
void main() {
    gl_FragColor = texture(video_texture, v_texture_coord);
}
