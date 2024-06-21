uniform sampler2D u_texture;    // Texture
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
#define PI 3.14159265

void main() {
      if(v_elevation>=-PI/6 && v_elevation<=0 && v_azimuth>=5*PI/9 && v_azimuth<=13*PI/18) {
         gl_FragColor = texture2D(u_texture, v_texcoord_fore);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}