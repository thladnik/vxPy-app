uniform float time;
uniform int polarity;
uniform float pos_azimut_angle;
uniform float pos_elev_angle;
uniform float movement_major_axis;  // radius of movement circle in degree
uniform float movement_minor_axis;
uniform float movement_angular_velocity;  // movement speed of dot
uniform float el_diameter_horiz;
uniform float el_diameter_vert;
uniform int el_rotating;
uniform int el_mirror;
uniform float el_color_r;
uniform float el_color_g;
uniform float el_color_b;
uniform float pos_azimut_offset; // horizontal offset of midline; in degree
varying vec3 v_position;
#define PI 3.14159265

// Transformation of coordinates. input in rad
vec3 sph2cart(in vec2 sph_coord){
    return vec3(sin(sph_coord.x) * cos(sph_coord.y),
                cos(sph_coord.x) * cos(sph_coord.y),
                sin(sph_coord.y));
}

// For Movement: Alligning ellipse axes to rotate arround movement center
vec3 rodrigues_rotation(vec3 vec, vec3 rot_ax, float alpha){
    return (1-cos(alpha)) * dot(vec, rot_ax) * rot_ax + cos(alpha) * vec + sin(alpha) * cross(rot_ax, vec);
}

// Ellipse Center on Path
vec3 get_pos_on_elliptical_path(vec3 movement_center_cart, float major_axis, float minor_axis, float angle){
    // Get axes of local plane in movement_center:
    vec3 north_pole = vec3(0.0, 0.0, 1.0);
    vec3 local_z_axis = movement_center_cart;
    vec3 local_x_axis = cross(north_pole, local_z_axis) / length(cross(north_pole, local_z_axis));
    vec3 local_y_axis = cross(local_z_axis, local_x_axis);

    // Ellipse point in local coordinates of local plane
    float local_x = (major_axis) * cos(angle);
    float local_y = (minor_axis) * sin(angle);
    vec3 local_coords = movement_center_cart + (local_x_axis * local_x + local_y_axis * local_y);

    return local_coords / length(local_coords);  // This is point on sphere
}


// Calculate Ellipse
vec3 make_ellipse(vec3 v_position, int el_rotating, float angle, vec3 ellipse_center, float major_axis, float minor_axis){
    // check if v_position and ellipse_center are not on opposite sides of sphere
    // in this case don't bother about calculating ellipse stuff
    float ellipse_eq_value = 0.0;
    if (distance(ellipse_center,v_position) > 1.0) {
        ellipse_eq_value = 1.0;
    }
    else {
        // Define local coordinate axes of tangent plane in ellipse_center.
        // New axes are all cross products of center ellipse and defined north pole
        vec3 north_pole = vec3(0.0, 0.0, 1.0);
        vec3 local_z_axis = ellipse_center;// z achse - ist richtungsvekrot des punktes, ist schon einheitsvektor!
        vec3 local_x_axis = cross(north_pole, local_z_axis) / length(cross(north_pole, local_z_axis));// kreuzprodukt z achse & centrum ellipse, einheitsvekrot machen
        vec3 local_y_axis = cross(local_z_axis, local_x_axis);// kreuzprodukt z achse & neue x achse, sind alles schon einheitsvekroten

        // in case dot should rotate with ellipse, then recalculate x and y local axis again. Rotation arround local_z_axis
        if (el_rotating == 1) {
            local_x_axis = rodrigues_rotation(local_x_axis, local_z_axis, angle);
            local_y_axis = rodrigues_rotation(local_y_axis, local_z_axis, angle);
        }

        // Old...
        // Transform point to local tangent plane coordinates
        vec3 local_coords = mat3(local_x_axis.x, local_y_axis.x, local_z_axis.x,
                                 local_x_axis.y, local_y_axis.y, local_z_axis.y,
                                 local_x_axis.z, local_y_axis.z, local_z_axis.z) * v_position;

        // Check if the point is within the ellipse
        ellipse_eq_value = pow((local_coords.x / minor_axis), 2.0) + pow((local_coords.y / major_axis), 2.0);

        // New approach... project dot on tangent plane via angle!
        // float angle_p = float(acos(dot(v_position, ellipse_center) / (normalize(v_position) * normalize(ellipse_center))));
        // vec2 v2d = vec2(1.0,tan(angle_p));
    }

    // construct ellipse paint....
    float ellipse = step(1.0, ellipse_eq_value);
    return vec3(1.0-ellipse);
}



void main(){
    // Parameter of Movement path
    vec3 mov_center = sph2cart(vec2((pos_azimut_angle + pos_azimut_offset) / 180.0 * PI, pos_elev_angle / 180 * PI));
    vec3 mov_center_mirror = sph2cart(vec2((-pos_azimut_angle + pos_azimut_offset) / 180.0 * PI, pos_elev_angle / 180 * PI));
    float mov_major_axis = sin(radians(movement_major_axis / 2.0)); //  axis of movements
    float mov_minor_axis = sin(radians(movement_minor_axis / 2.0));

    // actual ellipse paramteters
    vec3 color = vec3(el_color_r, el_color_g, el_color_b);
    vec3 ellipse1_center = get_pos_on_elliptical_path(mov_center, mov_major_axis, mov_minor_axis, radians(movement_angular_velocity)*time);
    vec3 ellipse2_center = get_pos_on_elliptical_path(mov_center_mirror, mov_major_axis, mov_minor_axis, radians(movement_angular_velocity)*time);
    //float major_axis = sin(radians(el_diameter_horiz));  // size of printed ellipse
    //float minor_axis = sin(radians(el_diameter_vert));
    float major_axis = radians(el_diameter_horiz / 2.0);
    float minor_axis = radians(el_diameter_vert / 2.0);

    // Make ellipses
    vec3 ellipse1 = make_ellipse(v_position, el_rotating, radians(movement_angular_velocity)*time, ellipse1_center, major_axis, minor_axis);
    vec3 ellipse2 = make_ellipse(v_position, el_rotating, radians(movement_angular_velocity)*time, ellipse2_center, major_axis, minor_axis);  // mirror ellipse
    //vec3 mov_center1 = make_ellipse(v_position, 0, radians(movement_angular_velocity)*time, mov_center, sin(radians(0.5)), sin(radians(0.5))); // controls for center of movement
    //vec3 mov_center2 = make_ellipse(v_position, 0, radians(movement_angular_velocity)*time, mov_center_mirror, sin(radians(0.5)), sin(radians(0.5)));

    // Get colors
    //vec3 paint = ellipse1 + ellipse2 + mov_center1 + mov_center2;
    vec3 paint = ellipse1;
    if(el_mirror == 1){
        paint = ellipse1 + ellipse2;
    }

    // Polarity of foreground-background. 1 = black background, 0 = white background
    gl_FragColor = vec4(paint * color, 1.0);
    if(polarity == 1){
        gl_FragColor = vec4((paint * -1 +1) * color, 1.0);
    }

}




