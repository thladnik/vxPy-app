vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }

mat4 rotamat_from_quat(vec4 quat) {

    float q0 = quat.x;
    float q1 = quat.y;
    float q2 = quat.z;
    float q3 = quat.w;

    return mat4(q0*q0+q1*q1-q2*q2-q3*q3, 2*q1*q2-2*q0*q3,         2*q1*q3+2*q0*q2,         0.0,
                2*q1*q2+2*q0*q3,         q0*q0-q1*q1+q2*q2-q3*q3, 2*q2*q3-2*q0*q1,         0.0,
                2*q1*q3-2*q0*q2,         2*q2*q3+2*q0*q1,         q0*q0-q1*q1-q2*q2+q3*q3, 0.0,
                0.0,                     0.0,                     0.0,                     1.0
    );

}

varying vec3 v_position;
varying vec4 v_rotation;
varying float v_triangle_center_az;
varying float v_triangle_center_el;
uniform float time;
uniform float noise_scale;
varying vec4 v_start_rotation;
#define PI 3.14159265
varying vec4 v_rotation_stationary;
varying vec4 v_foreground_rotation_start;

void main() {
    float deltaU = 0.3 * time;
    mat4 rotmat = rotamat_from_quat(v_rotation);
//    mat4 rotmat_foreground = rotamat_from_quat(v_start_rotation);
    mat4 rotmat_foreground = rotamat_from_quat(v_foreground_rotation_start);
    mat4 rotmat_background_stationary = rotamat_from_quat(v_rotation_stationary);

//    For debugging:
//    mat4 rotmat = mat4(1., 0., 0., 0.,
//                       0., 1., 0., 0.,
//                       0., 0., 1., 0.,
//                       0., 0., 0., 1. );

    vec3 position = (rotmat * vec4(v_position, 1.0)).xyz;
    vec3 fore_position_original = (rotmat_foreground * vec4(v_position, 1.0)).xyz;
    vec3 fore_position = vec3(fore_position_original.x+4.5, fore_position_original.y, fore_position_original.z);
    vec3 background_stationary_position = (rotmat_background_stationary * vec4(v_position, 1.0)).xyz;
//  position = normalize(position);
//  fore_position = normalize(fore_position);
    float brightness = snoise(position*noise_scale);
    float brightness_foreground = snoise(fore_position*noise_scale);
    float brightness_foreground_stationary = snoise(fore_position_original*noise_scale);
    float brightness_background_stationary = snoise(background_stationary_position*noise_scale);
//    vec4 color = vec4(vec3(brightness), 1.0);
    vec4 color = vec4(vec3(step(brightness, 0.0)), 1.0);
    vec4 color_foreground = vec4(vec3(step(brightness_foreground, 0.0)), 1.0);
    vec4 color_foreground_stationary = vec4(vec3(step(brightness_foreground_stationary, 0.0)), 1.0);
    vec4 color_background_stationary = vec4(vec3(step(brightness_background_stationary, 0.0)), 1.0);
    if(v_triangle_center_az >= -1.5707966  && v_triangle_center_az <= -1.2384055 && v_triangle_center_el <= -0.08800039 && v_triangle_center_el >= -0.45215960){
       gl_FragColor = color_foreground;
    } else {
       gl_FragColor = color_background_stationary;
    }
}