// Input
uniform float u_color;

// Main
void main()
{
    gl_FragColor = vec4(vec3(u_color), 1.0);
}