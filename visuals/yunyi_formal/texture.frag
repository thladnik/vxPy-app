varying float v_azimuth;
varying float v_elevation;
#define PI 3.14159265

void main() {
      if(v_elevation>=-PI/6 && v_elevation<=0 && v_azimuth>=5*PI/9 && v_azimuth<=13*PI/18) {
         gl_FragColor = vec4(0.0, 1.0, 0.0, 1.);
      } else {
         gl_FragColor = vec4(1.0, 1.0, 0.0, 1.);
      }
}