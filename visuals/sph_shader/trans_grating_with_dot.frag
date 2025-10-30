#define PI 3.14159265

uniform float u_stime;
uniform float u_spat_period;
uniform float u_ang_velocity;
uniform float u_period;
uniform float u_elv;
uniform float u_ang_size;

varying vec3 v_position;


vec3 sph2cart(in vec2 sph_coord) {
    return vec3(sin(sph_coord.x)*cos(sph_coord.y),
                cos(sph_coord.x)*cos(sph_coord.y),
                sin(sph_coord.y));
}


void main() {
    float grating_color = smoothstep(-0.1,0.1,sin(v_position.x*(180/u_spat_period)+u_stime*PI/180*u_ang_velocity));

    vec3 dotloc = sph2cart(vec2(fract(u_stime/u_period)*PI,u_elv/180*PI));
    float secLen = sin(u_ang_size/360*PI);
    float dot_color = 1.-smoothstep(secLen,secLen+0.01,distance(v_position,dotloc));
    if(dot_color > .5) {
        gl_FragColor = vec4(vec3(dot_color) ,1.);
    } else {
        gl_FragColor = vec4(vec3(grating_color) / 2.,1.);
    }
}
