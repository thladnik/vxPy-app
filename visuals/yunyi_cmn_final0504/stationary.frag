uniform sampler2D u_texture;    // Texture
uniform float time;
varying float v_azimuth;
varying float v_elevation;
varying vec2  v_texcoord;  // output
varying vec2 v_tile_center;
varying float v_tile_center_x;
varying float v_tile_center_y;
varying vec2 v_texture_start_coords;
varying vec2 v_texture_start_coords_new;
//varying vec2  new_texcoord_fore;
varying float v_triangle_num;
varying vec2  v_texcoord_fore;
varying vec2 vertices1;
varying vec2 vertices2;
varying vec2 vertices3;
varying vec2 vertices1_new;
varying vec2 vertices2_new;
varying vec2 vertices3_new;
varying float triangle_size;
varying vec2 v_texture_foreground_cord;
varying vec2 v_texture_foreground_cordinate2D;
#define PI 3.14159265

//vec2 cen2tri{
               //ct1 =  [v_tile_center.x - triangle_size / 2 * sqrt(3),  v_tile_center.y - triangle_size / 2]
               //ct2 =  [v_tile_center.x + triangle_size / 2 * sqrt(3),  v_tile_center.y - triangle_size / 2]
               //ct3 =  [v_tile_center.x, v_tile_center.y + triangle_size / 2 * sqrt(3)]
               //return
//}


void main() {
      //float deltaU = 0.1 * time;
      //vec2 vertices1_new = vec2((vertices1.x+deltaU), vertices1.y);
      //vec2 vertices2_new = vec2((vertices2.x+deltaU), vertices2.y);
      //vec2 vertices3_new = vec2((vertices3.x+deltaU), vertices3.y);
      //vec2 v_texture_start_coords_new = vec2(cen2tri(v_tile_center).x + deltaU, cen2tri(v_tile_center).y);
      //vec2 texcoord_fore = vec2((v_texcoord_fore.x-deltaU), v_texcoord_fore.y);
      //if(v_tile_center == [[-1.8749145], [-0.17747939]] || v_tile_center == [[-1.5707964], [-0.36486384]]) {
      //if(v_elevation>= -PI/6 && v_elevation<= 0 && v_azimuth>=-121.71747*PI/180 && v_azimuth<=-69.09484*PI/180) {
      //if((v_tile_center_x == -1.8749145  && v_tile_center_y == -0.17747939) || (v_tile_center_x == -1.5707964  && v_tile_center_y == -0.36486384)){
      //if(v_tile_center_x >= -1.8749145  && v_tile_center_x <= -1.5707964 && v_tile_center_y <= -0.17747939 && v_tile_center_y >= -0.36486384){
      //if(v_elevation>= -PI/6 && v_elevation<= 0 && v_azimuth>=-2.1243706 && v_azimuth<=-1.2059325) {
      //vec2 v_texture_new_coords = vec2(v_texture_start_coords.x + deltaU, v_texture_start_coords.y);
      //float y_new = float(vertices1.y, vertices2.y, vertices3.y)
      //vec2 texture_foreground_cordinate2D_move = vec2((v_texture_foreground_cordinate2D.x+deltaU), v_texture_foreground_cordinate2D.y);
      if(v_tile_center_x >= -1.9031870  && v_tile_center_x <= -1.2384055 && v_tile_center_y <= -0.08800039 && v_tile_center_y >= -0.45215960){
         gl_FragColor = texture2D(u_texture, v_texture_start_coords/2);
      } else {
         gl_FragColor = texture2D(u_texture, v_texcoord);
      }
}