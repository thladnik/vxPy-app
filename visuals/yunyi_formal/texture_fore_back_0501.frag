uniform sampler2D u_texture;    // Texture
uniform float time;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
//varying vec2  new_texcoord_fore;
#define PI 3.14159265

void main() {
      float deltaU = 0.05 * time;
      vec2 texcoord_fore = vec2((v_texcoord_fore.x+deltaU), v_texcoord_fore.y);
      if(v_elevation>=-PI/6 && v_elevation<=0 && v_azimuth>=4*PI/9 && v_azimuth<=11*PI/18) {
         gl_FragColor = texture2D(u_texture, texcoord_fore);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}