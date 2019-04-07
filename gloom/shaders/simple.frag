#version 430 core

out vec4 color;

uniform layout(location=0) vec2 window_size;
uniform layout(location=1) float time;
uniform layout(location=2) float bass_amplitude;

uniform layout(binding=0) sampler2D noise_texture;
uniform layout(binding=1) sampler2D planet_texture;

const int MARCHSTEPS = 400;
const float MIN_DIST = 0.0f;
const float MAX_DIST = 100.0f;
const float EPSILON = 0.0005;
const float PI = 3.1415926535897932384626433832795;

const vec3 underwater_color = vec3(0.0f, 0.0f, 0.10f);
const vec3 sunny_sky_color = vec3(0.31f, 0.62f, 0.86f);
const vec3 rainy_sky_color = vec3(0.22f, 0.26f, 0.29f);
const vec3 sphere_color = vec3(1.0f, 0.2f, 0.0f);
const vec3 periscope_color = vec3(0.0f, 0.0f, 0.0f);

float lightning_coeff = 0.0f;
float cloud_diffusion = 0.3f;
const float cloud_darkness = 0.2;
const float cloud_blend = 0.4f;

const float weather_strength = 0.10f;
const float waves_height = 1.0f;
const int waves_precision = 5;

vec3 scene_eye = vec3(10.0f, 1.0f, 5.0f);
vec3 global_light_pos = vec3(10.0f, 8.0f, -10.0f);

#define SPHERE_Y_TRANSLATION 1.0f
#define SPHERE_TRANSLATION (sd_translate(p, vec3(0.0f, SPHERE_Y_TRANSLATION, 0.0f)))
#define sd_sphere_call (sd_sphere(SPHERE_TRANSLATION, sphere_radius()))

#define PERISCOPE_Z_TRANSLATION (time/2 - 5.0f)
#define PERISCOPE_Y_TRANSLATION (-0.3f)
#define PERISCOPE_X_TRANSLATION (4.0f)
#define PERISCOPE_CYL_TRANSLATION (sd_translate(p, \
    vec3(PERISCOPE_X_TRANSLATION, PERISCOPE_Y_TRANSLATION - 0.42, PERISCOPE_Z_TRANSLATION)))
#define PERISCOPE_TOR_TRANSLATION (sd_translate(p, \
    vec3(PERISCOPE_X_TRANSLATION, PERISCOPE_Y_TRANSLATION + 0.15f, PERISCOPE_Z_TRANSLATION+0.05)))
#define PERISCOPE_WIN_TRANSLATION (sd_translate(p, \
    vec3(PERISCOPE_X_TRANSLATION, PERISCOPE_Y_TRANSLATION + 0.15f, PERISCOPE_Z_TRANSLATION+0.15)))
#define sd_periscope_call (sd_scene_periscope(p))

#define REFLECTION
#define REFLECTION_PERISCOPE
#define REFRACTION
// #define SHADOW
// #define AO
#define RAIN
#define RAIN_SPLASH
#define CLOUDS
#define PLANET
#define LIGHTNING
// #define MOVE_CAMERA

// https://stackoverflow.com/questions/10364575/normalization-in-variable-range-x-y-in-matlab.
float normalize_range(float f_min, float f_max, float t_min, float t_max, float x)
{
    float range = f_max - f_min;
    float range2 = t_max - t_min;

    float normalized_x = (x - f_min) / range;
    normalized_x = (normalized_x * range2) + t_min;

    return normalized_x;
}

// https://thebookofshaders.com/10/.
float random(float x)
{
    return fract(21654.6512 * sin(385.51 * x));
}

// https://thebookofshaders.com/10/.
float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 48301.231*sin(time/3));
}

// https://thebookofshaders.com/13/.
// Standard 2D noise function using Cubic Hermine.
float noise(vec2 st)
{
    vec2 i = floor(st);
    vec2 f = floor(st);

   // Create four corners of a 2D tile from random.
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Cubic Hermine Curve (smoothstep).
    f = f*f*(3.0-2.0*f);

    return mix(a, b, f.x) +
        (c - a)*f.y*(1.0 - f.x) +
        (d - b)*f.x*f.y;
}

// https://www.shadertoy.com/view/XdsGDB for an introduction to water.
// Another noise function that is used to generate the water with a
// Cubic Hermine Curve.
vec2 noise(vec3 p)
{
    vec3 i = floor(p);
    vec3 f = fract(p);

    // Cubic Hermine Curve (smoothstep).
    f = f*f*(3.0-2.0*f);

    // Until here, everything is similar to the noise function above
    // using vec2 <st>. However, here we use a loaded texture to
    // generate the noise, instead of an explicit function -- so we
    // just sample that texture.
    vec2 uv = (i.xy + vec2(37.0, 17.0)*i.z);
    vec4 rg = textureLod(noise_texture, (uv+f.xy+0.5)/256.0, 0.0);

    return mix(rg.yw, rg.xz, f.z);
}

// https://thebookofshaders.com/13/.
// Completely standard Fractal Brownian Motion.
float fbm(vec2 p)
{
    const int octaves = 5;
    float v = 0.0, a = 0.5;

    for (int i = 0; i < octaves; ++i) {
        v += noise(vec3(p, 1.0f)).y*a;

        p *= 2.0;
        a *= 0.5;
    }

    return v;
}

// See https://www.shadertoy.com/view/XdsGDB or
// https://www.shadertoy.com/view/Ms2SD1 for how octaves are used to
// create realistic water maps, much in the same vein as FBM is used
// to use superposition to create interesting waves. IQ also explains
// the method (animated FBM with something like a Perlin noise
// function) in
// https://iquilezles.org/www/articles/simplewater/simplewater.htm.
float sd_waves(vec3 p)
{
    float plane_height = p.y;
    p *= vec3(0.20);

    float f = 0.0;
    float amplitude = 2.0f;

    // The actual FBM. We sample our noise texture, along with the
    // superpositioning of several waves to create realistic looking
    // water. The more octaves that is used, the more realistic the
    // water will look (at the cost of computation, of course). I
    // thoroughly recommend having a look at
    // https://thebookofshaders.com/13/ for further insights on this.
    p += time*vec3(0, weather_strength, weather_strength);
    for (int i = 0; i < waves_precision; ++i) {
        // You don't necessarily _need_ to do this translation, the
        // waves will look alright without it. It does, however, make
        // for more "movement" within the motion of the waves.
        p = (p.yzx + p.zyx*vec3(1, -1, 1)) / sqrt(2.0);

        // It doesn't really matter _that_ much whether we use the .x
        // or .y of the 2-dimensional noise. We could even use the
        // magnitude of it. abs is used to create _sharper_ valleys.
        f += abs(noise(p).x - 0.5);

        f *= amplitude;
        p *= 2.0;
    }

    f /= exp2(float(waves_precision));
    float turbulence = (0.5 - f)*waves_height;

    return plane_height - turbulence;
}

// The old plane SD-function to create very simple planes displaced by
// a simple sine.
float sd_plane(vec3 p)
{
    float dist = length(p);

    if (dist > 7.5f)
        return p.y;

    return p.y + 0.03*(dist)*cos(dist*5.0f-time*5.0f);
}

// Calculate what the sphere of the radius should be, given a bass
// amplitude output from an FFT.
float sphere_radius()
{
    if (bass_amplitude < 0)
        return 0.90;

    return 0.45 + normalize_range(60.0f, 120.0f, 0.0f, 0.2f, bass_amplitude);
}

// From here there are a couple different SDF functions for different
// shapes. Inigo Quilez has written a lot of great articles on this on
// his blog:
// https://iquilezles.org/www/articles/distfunctions/distfunctions.htm. It
// is also possible to watch it in action at
// https://www.shadertoy.com/view/Xds3zN.
float sd_sphere(vec3 p, float r)
{
    return length(p) - r;
}

float sd_box(vec3 p, vec3 b, float r)
{
    vec3 d = abs(p) - b;
    return length(max(d, 0.0f)) - r;
}

// Vertical sylinder, arbitrary orientation is a bit more tricky.
float sd_cylinder(vec3 p, vec2 dim)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - dim;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0f));
}

float sd_torus(vec3 p, vec2 t)
{
    vec2 q = vec2(length(p.xz)- t.x, p.y);
    return length(q) - t.y;
}

float sd_intersect(float sda, float sdb)
{
    return max(sda, sdb);
}

float sd_union(float sd_a, float sd_b)
{
    return min(sd_a, sd_b);
}

// https://www.iquilezles.org/www/articles/smin/smin.htm. Gorgeous.
float sd_smooth_union(float sda, float sdb, float k)
{
    float h = clamp(0.5f + 0.5f*(sdb-sda)/k, 0.0f, 1.0f);
    return mix(sdb, sda, h) - k*h*(1.0f-h);
}

float sd_difference(float sda, float sdb)
{
    return max(sda, -sdb);
}

float sd_displace(vec3 p)
{
    return sin(45.0f*p.x) * sin(45.0f*p.y) * sin(45.0f*p.z);
}

vec4 sd_elongate(vec3 p, vec3 dim)
{
    vec3 q = abs(p) - dim;
    return vec4(max(q, 0.0f), min(max(q.x, max(q.y, q.z)), 0.0));
}

vec3 sd_translate(vec3 p, vec3 translation)
{
    return p - translation;
}

vec3 sd_rotateY(vec3 p, float theta)
{
    return (mat4(
        vec4(cos(theta), 0.0f, sin(theta), 0.0f),
        vec4(0.0f, 1.0f, 0.0f, 0.0f),
        vec4(-sin(theta), 0.0f, cos(theta), 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    ) * vec4(p, 1.0f)).xyz;
}

vec3 sd_rotateZ(vec3 p, float theta)
{
    return (mat4(
        vec4(cos(theta), -sin(theta), 0.0f, 0.0f),
        vec4(sin(theta), cos(theta), 0.0f, 0.0f),
        vec4(0.0f, 0.0f, 1.0f, 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    ) * vec4(p, 1.0f)).xyz;
}

float sd_scene_sphere(vec3 p)
{
    return sd_sphere_call;
}

float sd_scene_periscope(vec3 p)
{
    p.y += sin(time)/10.0f;

    float cylinder = sd_cylinder(PERISCOPE_CYL_TRANSLATION, vec2(0.07f, 0.50f));
    float window = sd_cylinder(PERISCOPE_WIN_TRANSLATION.xzy, vec2(0.1f, 0.001));

    vec3 p_torus = (PERISCOPE_TOR_TRANSLATION.xzy);
    vec4 torus_elongation = sd_elongate(p_torus,  vec3(0.0, 0.1, 0.0));
    float torus = min(1e10, torus_elongation.w + sd_torus(torus_elongation.xyz, vec2(0.1, 0.01)));

    float scene = sd_smooth_union(cylinder, torus, 0.05f);
    scene = sd_union(scene, window);
    return scene;
}

// This is required to be its own scene, as to be able to trace it
// alone to color it completely black.
float sd_scene_window(vec3 p)
{
    p.y += sin(time)/10.0f;

    return sd_cylinder(PERISCOPE_WIN_TRANSLATION.xzy, vec2(0.1f, 0.001));
}

float sd_scene_plane(vec3 p)
{
    return sd_waves(sd_translate(p, vec3(0.0f, -0.5f, 0.0f)));
}

float sd_scene(vec3 p)
{
    float plane = sd_waves(sd_translate(p, vec3(0.0f, -0.5f, 0.0f)));
    float sphere = sd_sphere_call;
    float periscope = sd_periscope_call;

    float un = sd_union(plane, sphere);
    un = sd_union(un, periscope);
    return un;
}

float trace_sphere(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_sphere(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

float trace_periscope(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_periscope(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

float trace_window(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_window(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

float trace_plane(vec3 eye, vec3 dir, float start, float end)
{
    float depth = start;

    for (int i = 0; i < MARCHSTEPS; ++i) {
        float dist = sd_scene_plane(eye + depth * dir);

        if (dist < EPSILON)
            return depth;

        depth += dist;
        if (depth >= end)
            return end;
    }

    return end;
}

vec3 ray_direction(float fov, vec2 size, vec2 fc)
{
    vec2 xy = fc - size / 2.0f;
    float z = size.y / tan(radians(fov) / 2.0f);
    return normalize(vec3(xy, -z));
}

vec3 est_normal(vec3 p)
{
    const float EPSILON = 0.01*length(p);
    return normalize(vec3(
          sd_scene(vec3(p.x + EPSILON, p.y, p.z)) - sd_scene(vec3(p.x - EPSILON, p.y, p.z)),
          sd_scene(vec3(p.x, p.y + EPSILON, p.z)) - sd_scene(vec3(p.x, p.y - EPSILON, p.z)),
          sd_scene(vec3(p.x, p.y, p.z + EPSILON)) - sd_scene(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

// We need an estimation for the sphere in particular for when we are
// doing transparent tracing (i.e. refraction in the water, otherwise
// we just get the normal of the water plane).
vec3 est_sphere_normal(vec3 p, float r)
{
    const float EPSILON = 0.001;
    p = SPHERE_TRANSLATION;
    return normalize(vec3(
        sd_sphere(vec3(p.x + EPSILON, p.y, p.z), r) - sd_sphere(vec3(p.x - EPSILON, p.y, p.z), r),
        sd_sphere(vec3(p.x, p.y + EPSILON, p.z), r) - sd_sphere(vec3(p.x, p.y - EPSILON, p.z), r),
        sd_sphere(vec3(p.x, p.y, p.z + EPSILON), r) - sd_sphere(vec3(p.x, p.y, p.z - EPSILON), r)
    ));
}

// http://delivery.acm.org/10.1145/1190000/1185834/p153-evans.pdf?ip=129.241.110.156&id=1185834&acc=ACTIVE%20SERVICE&key=CDADA77FFDD8BE08%2E5386D6A7D247483C%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1552050912_e7cfd8c9dad8342ae20b52e0aeabbaf2
// http://www.iquilezles.org/www/material/nvscene2008/rwwtt.pdf
float ambient_occlusion(vec3 pos, vec3 normal)
{
    float occ = 0.0f;
    float sca = 1.0f;

    for (int i = 0; i < 5; ++i) {
        float h = 0.001 + 0.15*float(i)/4.0f;
        float d = sd_scene(pos + h*normal);
        occ += (h-d)*sca;
        sca *= 0.95;
    }

    return clamp(1.0f - 1.5f*occ, 0.0f, 1.0f);
}

float penumbra_shadow(vec3 ro, vec3 rd, float mint, float maxt, float k)
{
    float EPSILON = 0.0001;
    float res = 1.0f;

    for (float t = mint; t < maxt;) {
        float h = sd_scene(ro + rd*t);
        if (h < EPSILON)
            return 0.0f;
        res = min(res, k*h/t);
        t += h;
    }

    return res;
}

vec3 phong_light_contrib(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                         vec3 light_pos, vec3 light_intensity, vec3 normal)
{
    vec3 N = normal;
    vec3 L = normalize(light_pos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dot_LN = dot(L, N);
    float dot_RV = dot(R, V);

    vec3 diffuse_light_intensity = light_intensity;
    light_intensity = vec3(1.0f, 1.0f, 1.0f);

    if (dot_LN < 0.0f)
        return vec3(0.0f, 0.0f, 0.0f);

    if (dot_RV < 0.0f)
        return diffuse_light_intensity * (k_d * dot_LN);

    float shadow = 1.0f;
    #ifdef SHADOW
    if (length(p) < 5.0f && eye == scene_eye)
        shadow = penumbra_shadow(p, L, 0.01f, 100.0f, 16.0f);
    #endif

    return diffuse_light_intensity * k_d * dot_LN * shadow
        + light_intensity * k_s * pow(dot_RV, alpha);
}

vec3 phong_illumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha,
                        vec3 p, vec3 eye, vec3 light_intensity, vec3 normal)
{
    float occlusion = 1.0f;
    #ifdef AO
    occlusion = ambient_occlusion(p, normal);
    #endif
    vec3 ambient_light = 0.3 * vec3(1.0f, 1.0f, 1.0f);
    vec3 tmp_color = k_a * ambient_light * occlusion;
    vec3 light_pos = global_light_pos;

    tmp_color += phong_light_contrib(k_d, k_s, alpha, p, eye,
                                     light_pos, light_intensity, normal);

    return tmp_color;
}

mat4 camera(vec3 eye, vec3 center, vec3 up)
{
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0f),
        vec4(u, 0.0f),
        vec4(-f, 0.0f),
        vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
}

vec3 shade_orb(vec3 orb_color, vec3 p_sphere)
{
    vec2 uv;

    // Pretty standard wrapping of a square texture around a
    // sphere.
    uv.x = atan(p_sphere.x, p_sphere.z)/(6.2831*2.0) - time*0.05;
    uv.y = (acos(p_sphere.y + SPHERE_Y_TRANSLATION*-1.0f)/3.1416)*0.5;


    vec3 planet_color = texture(planet_texture, uv.yx).xyz;

    orb_color = mix(orb_color,
                    vec3(1.0f, 1.0f, 0.0f),
                    smoothstep(0.5f, 1.0f, fbm(uv)));

    // Mix in ground level of the planet.
    orb_color = mix(orb_color,
                    ((vec3(0.0f) +
                      0.20*texture(planet_texture, 15.5*uv.yx).xyz)*0.4),
                    smoothstep(0.1, 0.2, planet_color.x));

    return orb_color;
}

vec4 shade_scene()
{
    vec3 k_a = vec3(0.5f, 0.5f, 0.5f);
    vec3 k_d = vec3(0.8f, 0.8f, 0.8f);
    vec3 k_s = vec3(0.4f, 0.4f, 0.4f);
    float shininess = 16.0f;

    vec3 eye = scene_eye;

    #ifdef MOVE_CAMERA
    eye = sd_rotateY(eye, time/20);
    #endif

    vec3 ray_dir = ray_direction(45.0f, window_size, gl_FragCoord.xy);
    mat4 camera_mat = camera(eye, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
    ray_dir = (camera_mat * vec4(ray_dir, 1.0f)).xyz;

    float dist_plane = trace_plane(eye, ray_dir, MIN_DIST, MAX_DIST);
    float dist_sphere = trace_sphere(eye, ray_dir, MIN_DIST, MAX_DIST);
    float dist_periscope = trace_periscope(eye, ray_dir, MIN_DIST, MAX_DIST);

    vec4 col = vec4(rainy_sky_color, 1.0f);

    #ifdef RAIN
    // https://www.shadertoy.com/view/XdSGDc. We create a sheet of
    // rain to overlay onto the image without caring about depth.
    vec2 q = gl_FragCoord.xy / window_size;
    float f = 10.0f;
    vec2 st = f * (q*vec2(1.5f, 0.05f) + vec2(-time*0.1 + q.y*0.5, time*0.12));
    f = (texture(noise_texture, st*0.5, -99.0).x + texture(noise_texture, st*0.284, -99.0).y);
    f = clamp(pow(abs(f)*0.5f, 29.0f)*140.0f, 0.00, q.y*0.4f + 0.5f);
    vec3 brightness = vec3(0.25f);
    vec4 rain = vec4(brightness*f, 1.0f);
    #endif

    #ifdef LIGHTNING
    // Create a seed to check whether the lightning is striking, such
    // that we can set the diffusion of the clouds differently to see
    // an acual lightning strike.
    float lightning_seed = random(vec2(1.0f, floor(time + (1.0 / 10.0))));
    if (lightning_seed >= 0.98) {
        cloud_diffusion = 0.7;
        lightning_coeff = 3.0f;
    } else {
        cloud_diffusion = 0.3;
        lightning_coeff = 0.0f;
    }
    #endif

    if (min(dist_plane, dist_sphere) > MAX_DIST - EPSILON) {
        // If we hit absolutely nothing, we are drawing the sky.
        #ifdef CLOUDS
        // https://www.shadertoy.com/view/Xds3Rj for the inspiration
        // for the clouds, changed to use our FBM and noise
        // functions. Clouds are explained in detail at
        // https://thebookofshaders.com/13/.
        vec2 uv = q*2.0 - 1.0;

        float cloud_fbm = fbm(uv - vec2(time / 10, 0.0));
        cloud_fbm = 1.0 - abs(cloud_fbm*2.0 - 1.0);
        vec3 cloud = pow(vec3(cloud_fbm), vec3(cloud_diffusion)) - (uv.y + 3.0)*cloud_darkness;

        #ifdef LIGHTNING
        cloud += vec3(1.0f) * lightning_coeff*pow(length(cloud), 2);
        #endif

        col = mix(col, vec4(cloud, 1.0f), cloud_blend);
        #endif


        #ifdef RAIN
        col += rain;
        #endif

        return col;
    }

    if (dist_periscope < dist_plane && dist_periscope < dist_sphere) {
        // When we hit the periscope.
        vec3 p_periscope = eye + dist_periscope*ray_dir;
        vec3 n_periscope = est_normal(p_periscope);

        if (abs(dist_periscope - trace_window(eye, ray_dir, MIN_DIST, MAX_DIST)) < 0.001)
            return vec4(0.0f, 0.0f, 0.0f, 1.0f);

        col = vec4(phong_illumination(k_a, k_d, k_s, 32.0f,
                                      p_periscope, eye, periscope_color,
                                      n_periscope), 1.0f);

        #ifdef REFLECTION_PERISCOPE
        vec3 refl_dir = reflect(ray_dir, n_periscope);
        float refl_dist_water = trace_plane(p_periscope, refl_dir, MIN_DIST, MAX_DIST);
        vec4 reflection = vec4(rainy_sky_color, 1.0f);
        if (refl_dist_water < MAX_DIST - EPSILON) {
            vec3 refl_p_water = p_periscope + refl_dist_water*refl_dir;
            vec3 refl_water_n = est_normal(refl_p_water);
            reflection = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                                 refl_p_water, p_periscope, underwater_color,
                                                 refl_water_n), 1.0f);
        }
        col = mix(col, reflection, 0.4f);
        #endif
     } else if (dist_plane < dist_sphere) {
        // When we hit the water.
        vec3 p_plane = eye + dist_plane*ray_dir;
        vec3 n_plane = est_normal(p_plane);

        // Standard color of water.
        col = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                      p_plane, eye, underwater_color,
                                      n_plane), 1.0f);

        #ifdef REFRACTION
        float raydotn = dot(ray_dir, n_plane);
        vec3 refr_ray = normalize(ray_dir + (-cos(1.10*acos(-raydotn))-raydotn)*n_plane);
        float refr_dist_sphere = trace_sphere(p_plane, refr_ray, MIN_DIST, MAX_DIST);
        if (refr_dist_sphere < MAX_DIST - EPSILON) {
            vec3 refr_p_sphere = p_plane + refr_dist_sphere*refr_ray;
            vec3 sphere_normal = est_sphere_normal(refr_p_sphere, sphere_radius());
            vec4 refr_color = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                                      refr_p_sphere, p_plane, sphere_color,
                                                      sphere_normal), 1.0f);
            col = mix(col, refr_color, exp(-refr_dist_sphere));
        }
        #endif

        #ifdef REFLECTION
        float fresnel = pow(1.0f-abs(dot(ray_dir, n_plane)), 2.0f);
        vec3 refl_dir = reflect(ray_dir, n_plane);
        float refl_dist_sphere = trace_sphere(p_plane, refl_dir, MIN_DIST, MAX_DIST);
        vec4 reflection = vec4(rainy_sky_color, 1.0f);
        if (refl_dist_sphere < MAX_DIST - EPSILON) {
            vec3 refl_p_sphere = p_plane + refl_dist_sphere*refl_dir;
            vec3 refl_sphere_n = est_sphere_normal(refl_p_sphere, sphere_radius());

            #ifdef PLANET
            reflection = vec4(shade_orb(sphere_color, refl_p_sphere), 1.0f);
            fresnel = pow(1.0f-abs(dot(ray_dir, n_plane)), 7.0f);
            #else
            reflection = vec4(phong_illumination(k_a, k_d, k_s, shininess,
                                                 refl_p_sphere, p_plane, sphere_color,
                                                 refl_sphere_n), 1.0f);
            #endif
        }

        col = mix(col, reflection, fresnel);
        #endif

        #ifdef RAIN_SPLASH
        vec2 q = (gl_FragCoord.xy / window_size)*500.0f;
        vec2 i = floor(q);
        vec2 f = fract(q);

        float splash = random(i);
        if (splash >= 0.999f)
            col = mix(col, vec4(1.0f), 0.5f);
        #endif
    } else {
        // When we hit the orb.
        vec3 orb_color = sphere_color;
        vec3 p_sphere = eye + dist_sphere*ray_dir;

        // Adapted from iqs very nice
        // https://www.shadertoy.com/view/4sf3Rn.
        #ifdef PLANET
        orb_color = shade_orb(orb_color, p_sphere);
        #endif

        col = vec4(orb_color, 1.0f);

        // Specular shade manually.
        vec3 N =  est_sphere_normal(p_sphere, sphere_radius());

        vec3 light_pos = global_light_pos;
        vec3 L = normalize(light_pos - p_sphere);
        vec3 V = normalize(eye - p_sphere);
        vec3 R = normalize(reflect(-L, N));
        float dot_RV = dot(R, V);

        // Use the red intensity to determine how much to specularly
        // light the orb.
        if (!(dot_RV < 0.0f))
            col += vec4(1.0f) * 0.5 * pow(dot_RV, 2.0f) * col.r;
    }

    #ifdef RAIN
    col += rain;
    #endif

    return col;
}

void main()
{
    vec4 scene_col = shade_scene();

    #ifdef LIGHTNING
    scene_col += vec4(0.05f) * lightning_coeff;
    #endif

    color = scene_col;
}
