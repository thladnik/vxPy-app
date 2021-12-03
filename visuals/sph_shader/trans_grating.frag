uniform float u_stime;
uniform float u_spat_period;
uniform float u_ang_velocity;
varying vec3 v_position;
#define PI 3.14159265

void main() {
    vec3 color = vec3(smoothstep(-0.1,0.1,sin(v_position.x*(180/u_spat_period)+u_stime*PI/180*u_ang_velocity)));
    gl_FragColor = vec4(color,1.);
}
