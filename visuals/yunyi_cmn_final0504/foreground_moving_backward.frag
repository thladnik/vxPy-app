uniform sampler2D u_texture;    // Texture
uniform float time;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 v_texture_start_coords;
//varying vec2  new_texcoord_fore;
#define PI 3.14159265

void main() {
      float deltaU = 0.1 * time;
      vec2 texture_start_coords = vec2((v_texture_start_coords.x+deltaU), v_texture_start_coords.y);
      if(v_tile_center_x >= -1.9031870  && v_tile_center_x <= -1.2384055 && v_tile_center_y <= -0.08800039 && v_tile_center_y >= -0.45215960){
         gl_FragColor = texture2D(u_texture, texture_start_coords/2);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}