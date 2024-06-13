uniform sampler2D u_texture;    // Texture
uniform float time;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2  v_texcoord_fore;
varying vec2  v_texture_start_coords;
varying vec2 v_texture_foreground_cordinate2D;
varying vec2 v_tile_orientation;
//varying vec2  new_texcoord_fore;
varying float v_tile_center_x;
varying float v_tile_center_y;
//varying vec2 v_tile_center;
varying vec2 vertices1;
varying vec2 vertices2;
varying vec2 vertices3;
varying vec2 v_texture_foreground_cord;
varying float v_triangle_angle;
#define PI 3.14159265


void main() {
      float deltaU = 0.1 * time;
      mat3 matrix_time = mat3(
                         1, 0, 0,
                         0, 1, 0,
                         0.1 * time, 0, 1);
      vec3 v_texture_move_coords= vec3(vec2(v_texture_start_coords), 0);
      vec2 v_texcoord_fore = vec2((v_texcoord_fore.x+deltaU), v_texcoord_fore.y);
      vec2 vertices1_new = vec2((vertices1.x+deltaU), vertices1.y);
      vec2 vertices2_new = vec2((vertices2.x+deltaU), vertices2.y);
      vec2 vertices3_new = vec2((vertices3.x+deltaU), vertices3.y);
      float distance_x = v_triangle_angle/sqrt(1+v_triangle_angle*v_triangle_angle);
      float distance_y = 1/sqrt(1+v_triangle_angle*v_triangle_angle);
      //mat3*2 matrix_foreground = mat3*2((vertices1_new.x, vertices2_new.x, vertices3_new.x), (vertices1_new.y, vertices2_new.y, vertices3_new.y));
      float verticle[3] = float[3] (vertices1_new.y, vertices2_new.y, vertices3_new.y);
      float horizontal[3] = float[3] (vertices1_new.x, vertices2_new.x, vertices3_new.x);
      struct texture_move_coords {float horizontal; float verticle;} texture_translate_coords;
      //float verticle[3] = float[3] (vertices1_new.y, vertices2_new.y, vertices3_new.y);
      //vec2 v_texture_start_coords_new = vec2(vertices1_new,  vertices2_new, vertices3_new);
      //vec2 texture_start_coords = vec2((v_texture_start_coords.x+deltaU*distance_x), (v_texture_start_coords.y-deltaU*distance_y));
      //vec2 texture_start_coords = vec2((v_texture_start_coords.x+deltaU), v_texture_start_coords.y);
     //vec2 texture_translate_coords = vec2(texture_move_coords(horizontal), texture_move_coords(verticle))/2;
      //vec3 texture_translate_coords = matrix_time * v_texture_move_coords;
      //vec2 texture_move_coords = vec2(texture_translate_coords.x, texture_translate_coords.y);
      float x = texture_translate_coords.horizontal;
      float y = texture_translate_coords.verticle;
      vec2 texture_moving_coords = vec2(x, y);
      float size = v_tile_center_x*v_tile_center_y*16/sqrt(3);
      vec2 v_tile_center = vec2(v_tile_center_x*size+deltaU, v_tile_center_y*size);
      vec2 texture_foreground_cordinate2D_move = vec2((v_texture_foreground_cordinate2D.x+deltaU), v_texture_foreground_cordinate2D.y);
      if(v_tile_center_x >= -1.9031870  && v_tile_center_x <= -1.2384055 && v_tile_center_y <= -0.08800039 && v_tile_center_y >= -0.45215960){
         gl_FragColor = texture2D(u_texture, texture_foreground_cordinate2D_move/2);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}


