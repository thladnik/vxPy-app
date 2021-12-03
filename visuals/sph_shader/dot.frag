uniform float u_stime;
uniform float u_period;
uniform float u_elv;
uniform float u_ang_size;
varying vec3 v_position;
#define PI 3.14159265


vec3 sph2cart(in vec2 sph_coord)
{
    return vec3(sin(sph_coord.x)*cos(sph_coord.y),
    cos(sph_coord.x)*cos(sph_coord.y),
    sin(sph_coord.y));
}

void main() {
    vec3 dotloc = sph2cart(vec2(fract(u_stime/u_period)*PI,u_elv/180*PI));
    float secLen = sin(u_ang_size/360*PI);
    vec3 color = vec3(1.-smoothstep(secLen,secLen+0.01,distance(v_position,dotloc)));
    gl_FragColor = vec4(color,1.);
}
