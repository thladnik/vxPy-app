// Input
attribute vec3  a_position;

// Main
void main()
{
    gl_Position = transform_position(a_position);
}