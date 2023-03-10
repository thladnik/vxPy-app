// Input
uniform float u_color;         // Interpolated fragment color (in)

// Do stuff
void main()
{
    gl_FragColor = vec4(vec3(u_color), 1.0);
}