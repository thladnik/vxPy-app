uniform sampler2D u_texture;    // Texture
uniform float time;
varying vec3  v_tile_center;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
//varying vec2  new_texcoord_fore;
#define PI 3.14159265

void main() {
      float deltaU = 0.1 * time;
      vec2 texcoord_fore = vec2((v_texcoord_fore.x-deltaU), v_texcoord_fore.y);
      if(v_elevation/ (2 * PI) * 360>= -30 && v_elevation<= 0 v_azimuth>=-121.71747 && v_azimuth<=-69.09484) {
         gl_FragColor = texture2D(u_texture, v_tile_center);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}