precision highp float;

varying vec2 v_texCoord;

uniform sampler2D u_image;
uniform sampler2D u_depth;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform vec2 u_imageResolution;

// Use lil-gui to control these!
uniform float u_parallax_strength;

uniform vec2 u_light_pos;

// --- 光沢用のUniform変数を追加 ---
uniform float u_specular_strength; // 光沢の強さ
uniform float u_shininess;         // 光沢の鋭さ
uniform float u_light_z;           // 光源の高さ

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

vec2 depth_normal(vec2 p){
    float depth = texture2D(u_depth, p).r;
    // 隣接ピクセルとの深度差から勾配を計算し、擬似的な法線とする
    return vec2(
        (texture2D(u_depth, p + vec2(0.04, 0.0)).r - depth) * u_resolution.x,
        (texture2D(u_depth, p + vec2(0.0, 0.04)).r - depth) * u_resolution.y
    );
}

void main() {
    float depth = texture2D(u_depth, v_texCoord).r;

    // 1. Parallax
    vec2 parallax_offset = (u_mouse - 0.5) * -1.0;
    vec2 parallaxCoord = v_texCoord + parallax_offset * depth * u_parallax_strength;
    
    // パララックス適用後の座標でベースカラーを取得
    vec3 color = texture2D(u_image, parallaxCoord).rgb;

    // --- ここから光沢（スペキュラーハイライト）の計算 ---

    // 2. 法線の計算
    // パララックス適用後の座標を使うことで、光沢も立体的に動く
    vec2 normal_2d = depth_normal(parallaxCoord);
    vec3 normal = normalize(vec3(normal_2d.x, normal_2d.y, 1.0));

    vec3 fragPos = vec3(parallaxCoord, depth);              // ピクセルの3D位置
    vec3 lightPos = vec3(u_light_pos, u_light_z);           // 光源の3D位置
    vec3 lightDir = normalize(lightPos - fragPos);          // 光線ベクトル
    vec3 viewDir = vec3(0.0, 0.0, 1.0);                     // 視線ベクトル (Z+方向から見ていると仮定)

    float aspect = u_imageResolution.x / u_imageResolution.y;
    lightDir.xy = (u_mouse - fragPos) * vec2(aspect, 1.0);

    vec3 halfwayDir = normalize(lightDir + viewDir);        // 半角ベクトル
    float spec_intensity = pow(max(dot(normal, halfwayDir), 0.0), u_shininess);
    vec3 specular = vec3(1.0) * spec_intensity * u_specular_strength;
    color += specular;

    gl_FragColor = vec4(color, 1.0);
}