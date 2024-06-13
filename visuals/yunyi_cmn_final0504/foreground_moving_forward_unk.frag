


uniform sampler2D u_texture;    // Texture
uniform float time;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
varying vec2  v_texture_start_coords;
//varying vec2  new_texcoord_fore;
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 vertices1;
varying vec2 vertices2;
varying vec2 vertices3;
#define PI 3.14159265

void main() {
      float deltaU = 0.1 * time;
      vec2 vertices1_new = vec2((vertices1.x+deltaU), vertices1.y);
      vec2 vertices2_new = vec2((vertices2.x+deltaU), vertices2.y);
      vec2 vertices3_new = vec2((vertices3.x+deltaU), vertices3.y);
      //vec2 v_texture_start_coords_new = vec2(vertices1_new,  vertices2_new, vertices3_new);
      vec2 texture_start_coords = vec2((v_texture_start_coords.x+deltaU), v_texture_start_coords.y);
      if(v_tile_center_x >= -1.9031870  && v_tile_center_x <= -1.2384055 && v_tile_center_y <= -0.08800039 && v_tile_center_y >= -0.45215960){
         //gl_FragColor = texture2D(u_texture, texture_start_coords/2);
         //cju
         //gl_FragColor=vec4(0,1,0,1);
         float sphere_u = (1.0+v_azimuth/PI)/2.0 + deltaU;
         float sphere_v = (1.0+v_elevation*2.0/PI)/2.0;
         vec2 uv_texcoord = vec2(sphere_u, sphere_v);
         gl_FragColor = texture2D(u_texture, uv_texcoord);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
         //cju
         //float sphere_u = (1.0+v_azimuth/PI)/2.0 + deltaU;
         //float sphere_v = (1.0+v_elevation*2.0/PI)/2.0;
         //vec2 uv_texcoord = vec2(sphere_u, sphere_v);
         //gl_FragColor = texture2D(u_texture, uv_texcoord);
      }
}