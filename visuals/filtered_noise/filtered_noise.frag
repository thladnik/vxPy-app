uniform sampler2D u_texture;
uniform float u_min_value;
uniform float u_max_value;
varying vec2 v_texcoord;

void main()
{
    vec4 colour = texture2D(u_texture, v_texcoord);
    float mappedValue = (colour.r - u_min_value) / (u_max_value - u_min_value);
    gl_FragColor = vec4(mappedValue, mappedValue, mappedValue, 1.0);
}