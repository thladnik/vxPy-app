// Input
uniform float luminance;

// Main
void main()
{
    gl_FragColor = vec4(vec3(luminance), 1.0);
}