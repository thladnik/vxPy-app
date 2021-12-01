uniform float u_stime;
uniform float u_spat_period;
uniform float u_ang_velocity;
varying vec3 v_position;
#define PI 3.14159265
float atan2(in float y, in float x)
{
    float norm = length(vec2(x,y));
    x /= norm;
    return mix(acos(x),-acos(x),int(y>0));
}

vec2 cart2sph(in vec3 cart_coord)
{
    cart_coord /= length(cart_coord); // normalization
    return vec2(atan2(cart_coord.y,cart_coord.x),atan2(cart_coord.z,length(cart_coord.xy)));
}

void main() {
    vec2 uv_pos = cart2sph(v_position.zxy);
    vec3 color = vec3(smoothstep(-0.1,0.1,sin(uv_pos.x*(180/u_spat_period)+u_stime*PI/180*u_ang_velocity)));
    gl_FragColor = vec4(color,1.);
}
