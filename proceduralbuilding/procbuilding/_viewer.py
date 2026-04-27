"""Standalone Panda3D viewer — also exposed as the `procbuilding-viewer` CLI."""
import argparse
import math

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AntialiasAttrib,
    LPoint3f,
    LVector3f,
    NodePath,
    Shader,
    TextureStage,
)

import procbuilding
from procbuilding import LShapedHouseParams, PolygonHouseParams, ResidentialHouseParams, RoofType
from procbuilding.geometry.builder import GeomBuilder

_CLAY_COLOR   = (0.72, 0.72, 0.72, 1.0)
_GROUND_COLOR = (0.12, 0.12, 0.12, 1.0)
_BG_COLOR     = (0.18, 0.18, 0.18, 1.0)

_INITIAL_ELEVATION = 28.0

# Normal-map TextureStage (sort=1 → p3d_Texture1 in shader)
_NOR_STAGE = TextureStage("nor")
_NOR_STAGE.setSort(1)

# Material variant pools — one from each list is chosen randomly per building load.
_PLASTER_KINDS    = ["plaster", "plaster_painted"]
_ROOF_KINDS       = ["roof_tile", "roof_slate", "roof_corrugated"]
_WOOD_KINDS       = ["wood", "wood_planks"]

# ---------------------------------------------------------------------------
# GLSL shaders
#
# Clay mode  : u_textured=0 → vertex color only, no texture sampler used.
# Textured   : u_textured=1 → diffuse from p3d_Texture0 (sRGB-decoded by GPU).
#              u_use_normals=1 → perturb normal from p3d_Texture1 (linear).
#
# Lighting model:
#   • Hemisphere ambient (sky/ground)
#   • 3 directional lights (key/fill/rim) in world space
#   • Derivative-based TBN normal perturbation (no tangent attribute needed)
#   • Ground-contact ambient occlusion (smooth darkening near Z=0)
#   • Height gradient (subtle top-brighter effect)
#   • Reinhard tone mapping + gamma encode for sRGB display
# ---------------------------------------------------------------------------

_VERT = """
#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

out vec3 v_normal_world;
out vec3 v_pos_vs;
out vec3 v_pos_ws;
out float v_world_z;
out vec4 v_color;
out vec2 v_uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vec4 pos_vs = p3d_ModelViewMatrix * p3d_Vertex;
    v_pos_vs       = pos_vs.xyz;
    v_pos_ws       = p3d_Vertex.xyz;   // model == world (buildings only translate, never rotate)
    v_normal_world = normalize(p3d_Normal);
    v_world_z      = p3d_Vertex.z;
    v_color        = p3d_Color;
    v_uv           = p3d_MultiTexCoord0;
}
"""

_FRAG = """
#version 130

in vec3 v_normal_world;
in vec3 v_pos_vs;
in vec3 v_pos_ws;
in float v_world_z;
in vec4 v_color;
in vec2 v_uv;

out vec4 p3d_FragColor;

uniform sampler2D p3d_Texture0;   // diffuse  (sRGB-decoded by GPU via F_srgb)
uniform sampler2D p3d_Texture1;   // normal map (linear, OpenGL tangent-space)
uniform mat3 p3d_NormalMatrix;    // rotates world-space normals to view space
uniform float u_textured;         // 1.0 = textured, 0.0 = clay
uniform float u_use_normals;      // 1.0 = apply normal map
uniform float u_building_height;
uniform float u_glass;            // 1.0 = window glass surface

// World-space directional lights
const vec3 KEY_DIR    = vec3(-0.408, -0.408,  0.816);
const vec3 KEY_COLOR  = vec3( 0.90,   0.85,   0.75 );
const vec3 FILL_DIR   = vec3( 0.577,  0.577, -0.577);
const vec3 FILL_COLOR = vec3( 0.15,   0.20,   0.38 );
const vec3 RIM_DIR    = vec3( 0.000,  0.707, -0.707);
const vec3 RIM_COLOR  = vec3( 0.22,   0.22,   0.22 );

// Hemisphere ambient
const vec3 SKY_AMB    = vec3(0.20, 0.22, 0.28);
const vec3 GROUND_AMB = vec3(0.18, 0.14, 0.10);

// --- Stochastic tiling ---
// Breaks visible texture repetition by hashing each tile cell to a random UV offset,
// then blending the 4 nearest offset samples with smoothstep weights.
vec2 hashf(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.xx + p.yx) * p.yx);
}

vec3 stoch(sampler2D s, vec2 uv) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);
    vec2 w = smoothstep(0.2, 0.8, f);
    vec3 c00 = texture(s, uv + hashf(i + vec2(0.0, 0.0))).rgb;
    vec3 c10 = texture(s, uv + hashf(i + vec2(1.0, 0.0))).rgb;
    vec3 c01 = texture(s, uv + hashf(i + vec2(0.0, 1.0))).rgb;
    vec3 c11 = texture(s, uv + hashf(i + vec2(1.0, 1.0))).rgb;
    return mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
}

void main() {
    vec3 Nw = normalize(v_normal_world);

    // --- Hemisphere ambient (used by both glass and opaque paths) ---
    vec3 ambient = mix(GROUND_AMB, SKY_AMB, 0.5 + 0.5 * Nw.z);

    // --- Glass material (early return) ---
    // Opaque reflective surface: Fresnel-blended environment reflection + specular.
    // No transparency — fully opaque so the interior is never visible.
    if (u_glass > 0.5) {
        vec3 Nv_g   = normalize(p3d_NormalMatrix * Nw);
        vec3 V_v    = normalize(-v_pos_vs);
        float NdotV = clamp(dot(Nv_g, V_v), 0.0, 1.0);

        // Schlick Fresnel: grazing angles reflect the environment strongly
        float F = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);

        // Reflect view direction back to world space for environment sampling
        vec3 R_vs = reflect(-V_v, Nv_g);
        vec3 R_ws = transpose(p3d_NormalMatrix) * R_vs;
        vec3 env  = mix(GROUND_AMB, SKY_AMB, 0.5 + 0.5 * normalize(R_ws).z);

        // Blinn-Phong specular from key light
        vec3 key_vs = normalize(p3d_NormalMatrix * KEY_DIR);
        vec3 H      = normalize(key_vs + V_v);
        float spec  = pow(max(dot(Nv_g, H), 0.0), 128.0) * 2.0;

        // Dark tint at normal incidence, environment at grazing — fully opaque
        vec3 glass_tint = vec3(0.04, 0.06, 0.10);
        vec3 glass_lit  = mix(glass_tint, env, F) + KEY_COLOR * spec;

        glass_lit = glass_lit / (glass_lit + vec3(1.0));
        glass_lit = pow(max(glass_lit, vec3(0.0)), vec3(1.0 / 2.2));
        p3d_FragColor = vec4(glass_lit, 1.0);
        return;
    }

    // --- Normal map perturbation (derivative-based TBN, world space) ---
    // TBN is built from world-space derivatives. Since buildings never rotate,
    // model space == world space, so v_pos_ws derivatives give world-space T/B.
    if (u_use_normals > 0.5) {
        vec3 dp1_ws = dFdx(v_pos_ws);
        vec3 dp2_ws = dFdy(v_pos_ws);
        vec2 duv1   = dFdx(v_uv);
        vec2 duv2   = dFdy(v_uv);
        float det   = duv1.x * duv2.y - duv1.y * duv2.x;
        if (abs(det) > 1e-6) {
            vec3 T_ws = normalize( duv2.y * dp1_ws - duv1.y * dp2_ws);
            vec3 B_ws = normalize(-duv2.x * dp1_ws + duv1.x * dp2_ws);
            mat3 TBN  = mat3(T_ws, B_ws, Nw);
            vec3 tn   = stoch(p3d_Texture1, v_uv) * 2.0 - 1.0;
            Nw = normalize(TBN * tn);
        }
    }

    // --- Directional diffuse ---
    float dk = max(dot(Nw, KEY_DIR),  0.0);
    float df = max(dot(Nw, FILL_DIR), 0.0);
    float dr = max(dot(Nw, RIM_DIR),  0.0);
    vec3 diffuse = KEY_COLOR * dk + FILL_COLOR * df + RIM_COLOR * dr;

    // --- Ground-contact AO (shadows base of walls) ---
    float ao = mix(0.55, 1.0, smoothstep(0.0, 0.5, v_world_z));

    // --- Height gradient (subtle top-brighter artistic effect) ---
    float ht       = clamp(v_world_z / max(u_building_height, 0.1), 0.0, 1.0);
    float height_f = mix(0.88, 1.0, ht);

    // --- Base colour ---
    vec3 base;
    if (u_textured > 0.5) {
        // stoch() breaks repetition; p3d_Texture0 F_srgb delivers linear texels
        vec3 tex = stoch(p3d_Texture0, v_uv);
        base = tex * v_color.rgb;
    } else {
        base = v_color.rgb;
    }

    vec3 lit = base * (ambient + diffuse) * ao * height_f;

    // --- Reinhard tone mapping ---
    lit = lit / (lit + vec3(1.0));

    // --- Gamma encode for sRGB framebuffer ---
    lit = pow(max(lit, vec3(0.0)), vec3(1.0 / 2.2));

    p3d_FragColor = vec4(lit, v_color.a);
}
"""


class BuildingViewer(ShowBase):
    """Clay/textured Panda3D viewer with GLSL lighting and orbit camera."""

    def __init__(self, building_type: str, params, clay_mode: bool = False) -> None:
        super().__init__()

        self.render.setAntialias(AntialiasAttrib.MMultisample, 4)

        self._building_type = building_type
        self._building_np: NodePath | None = None
        self._ground_np:   NodePath | None = None
        self._textured = not clay_mode

        # Material variants — randomised each time a building loads
        self._mat_plaster = "plaster"
        self._mat_roof    = "roof_tile"
        self._mat_wood    = "wood"

        self._shader = Shader.make(Shader.SL_GLSL, _VERT, _FRAG)
        self.render.setShader(self._shader)
        self.render.setShaderInput("u_building_height", 9.0)
        self.render.setShaderInput("u_textured",    1.0 if self._textured else 0.0)
        self.render.setShaderInput("u_use_normals", 1.0 if self._textured else 0.0)
        self.render.setShaderInput("u_glass",       0.0)

        self._add_ground_plane()

        self.disableMouse()
        self._orbit_angle = -25.0
        self._orbit_elevation = _INITIAL_ELEVATION
        self._orbit_radius = 22.0
        self._look_at = LPoint3f(0, 0, 3)

        self._load_building(params)

        self.setBackgroundColor(*_BG_COLOR)
        self.accept("q", self.userExit)
        self.accept("escape", self.userExit)
        self.accept("r", self._regen)
        self.accept("t", self._toggle_texture)
        self._setup_orbit()

        mode_str = "textured" if self._textured else "clay"
        print("=" * 52)
        print(f"  procbuilding viewer  [{mode_str}]")
        print("  Left / Right : orbit        Up / Down : elevation")
        print("  R            : random building")
        print("  T            : toggle clay / textured")
        print("  Q / Escape   : quit")
        print("=" * 52)

    # ------------------------------------------------------------------
    # Building management
    # ------------------------------------------------------------------

    def _randomise_materials(self) -> None:
        import random
        self._mat_plaster = random.choice(_PLASTER_KINDS)
        self._mat_roof    = random.choice(_ROOF_KINDS)
        self._mat_wood    = random.choice(_WOOD_KINDS)

    def _node_texture_map(self) -> dict[str, str]:
        return {
            "wall_":     self._mat_plaster,
            "edge_wall": self._mat_plaster,
            "chimney":   "brick",
            "slab_":     "concrete",
            "roof":      self._mat_roof,
            "balcony":   self._mat_plaster,
            "ac_unit":   "metal",
            "door":      self._mat_wood,
        }

    def _load_building(self, params) -> None:
        self._randomise_materials()
        if self._building_np is not None:
            self._building_np.removeNode()
        self._building_np = procbuilding.build(
            self._building_type, params=params, parent=self.render
        )
        self._apply_building_visuals()

        if isinstance(params, PolygonHouseParams):
            xs = [v[0] for v in params.verts]
            ys = [v[1] for v in params.verts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            building_height = params.num_floors * params.floor_height
            look_z = building_height * 0.45
            orbit_r = max(max(xs) - min(xs), max(ys) - min(ys)) * 2.2
        elif isinstance(params, LShapedHouseParams):
            cx = params.main_width / 2
            cy = params.main_depth / 2
            building_height = params.num_floors * params.floor_height
            look_z = building_height * 0.45
            orbit_r = max(params.main_width, params.main_depth) * 2.2
        else:
            z_eave = params.num_floors * params.floor_height
            building_height = z_eave + params.roof_pitch * (params.depth / 2)
            cx = params.width / 2
            cy = params.depth / 2
            look_z = params.num_floors * params.floor_height * 0.45
            orbit_r = max(params.width, params.depth) * 2.2

        self._building_np.setShaderInput("u_building_height", building_height)
        self._building_np.setPos(-cx, -cy, 0)
        self._orbit_radius = orbit_r
        self._look_at = LPoint3f(0, 0, look_z)
        self._update_camera()

    def _apply_building_visuals(self) -> None:
        np = self._building_np
        if self._textured:
            np.setColor(1.0, 1.0, 1.0, 1.0)
            self._apply_textures(np)
        else:
            np.setColor(*_CLAY_COLOR)
            np.clearTexture()
        self._apply_glass_nodes(np)

    def _apply_ground_visuals(self) -> None:
        if self._ground_np is None:
            return
        if self._textured:
            from procbuilding.textures import get_texture, get_normal_texture
            self._ground_np.setColor(1.0, 1.0, 1.0, 1.0)
            self._ground_np.setTexture(get_texture("cobblestone"), 1)
            self._ground_np.setTexture(_NOR_STAGE, get_normal_texture("cobblestone"))
        else:
            self._ground_np.setColor(*_GROUND_COLOR)
            self._ground_np.clearTexture()

    def _apply_textures(self, np: NodePath) -> None:
        from procbuilding.textures import get_texture, get_normal_texture
        node_tex = self._node_texture_map()
        for child in np.findAllMatches("**"):
            name = child.getName()
            for prefix, kind in node_tex.items():
                if name.startswith(prefix):
                    child.setTexture(get_texture(kind), 1)
                    child.setTexture(_NOR_STAGE, get_normal_texture(kind))
                    break

    def _apply_glass_nodes(self, np: NodePath) -> None:
        for child in np.findAllMatches("**/glass"):
            child.setShaderInput("u_glass", 1.0)
            child.setShaderInput("u_textured", 0.0)
            child.setShaderInput("u_use_normals", 0.0)

    def _toggle_texture(self) -> None:
        self._textured = not self._textured
        self.render.setShaderInput("u_textured",    1.0 if self._textured else 0.0)
        self.render.setShaderInput("u_use_normals", 1.0 if self._textured else 0.0)
        self._apply_building_visuals()
        self._apply_ground_visuals()
        print(f"  mode → {'textured' if self._textured else 'clay'}")

    def _regen(self) -> None:
        import random as _rng
        choice = _rng.choices(
            ["residential_house", "l_shaped_house"],
            weights=[0.55, 0.45],
        )[0]
        self._building_type = choice

        if choice == "l_shaped_house":
            params = LShapedHouseParams.random()
            info = (
                f"L-shape  {params.num_floors}F  "
                f"{params.main_width:.0f}×{params.main_depth:.0f}m  "
                f"notch {params.notch_width:.0f}×{params.notch_depth:.0f}m"
            )
        else:
            params = ResidentialHouseParams.random()
            info = (
                f"rect  {params.num_floors}F  "
                f"{params.width:.0f}×{params.depth:.0f}m  "
                f"{params.roof_type.name.lower()} roof"
            )
        self._load_building(params)
        print(f"  random: {info}")

    # ------------------------------------------------------------------
    # Ground plane
    # ------------------------------------------------------------------

    def _add_ground_plane(self) -> None:
        s = 120.0
        t = 0.6   # cobblestone tile size in metres (~60 cm per stone cluster)
        b = GeomBuilder("ground")
        b.add_quad(
            [LPoint3f(-s, -s, 0), LPoint3f(s, -s, 0),
             LPoint3f(s,   s, 0), LPoint3f(-s,  s, 0)],
            LVector3f(0, 0, 1),
            _GROUND_COLOR,
            uvs=[(-s/t, -s/t), (s/t, -s/t), (s/t, s/t), (-s/t, s/t)],
        )
        self._ground_np = self.render.attachNewNode(b.build())
        self._ground_np.setZ(-0.01)
        self._apply_ground_visuals()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera(self) -> None:
        angle_rad = math.radians(self._orbit_angle)
        elev_rad  = math.radians(self._orbit_elevation)
        r         = self._orbit_radius
        r_h = r * math.cos(elev_rad)
        r_v = r * math.sin(elev_rad)
        self.camera.setPos(LPoint3f(
            r_h * math.sin(angle_rad),
            -r_h * math.cos(angle_rad),
            r_v,
        ))
        self.camera.lookAt(self._look_at)

    # ------------------------------------------------------------------
    # Orbit input
    # ------------------------------------------------------------------

    def _setup_orbit(self) -> None:
        self._keys: dict[str, bool] = {
            "left": False, "right": False, "up": False, "down": False,
        }
        for key in self._keys:
            self.accept(f"arrow_{key}", self._key_down, [key])
            self.accept(f"arrow_{key}-up", self._key_up, [key])
        self.taskMgr.add(self._orbit_task, "orbit_task")

    def _key_down(self, key: str) -> None:
        self._keys[key] = True

    def _key_up(self, key: str) -> None:
        self._keys[key] = False

    def _orbit_task(self, task):
        dt = globalClock.getDt()
        if self._keys["left"]:
            self._orbit_angle -= 60 * dt
        if self._keys["right"]:
            self._orbit_angle += 60 * dt
        if self._keys["up"]:
            self._orbit_elevation = min(85, self._orbit_elevation + 40 * dt)
        if self._keys["down"]:
            self._orbit_elevation = max(5,  self._orbit_elevation - 40 * dt)
        self._update_camera()
        return task.cont


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Procedural building viewer")
    parser.add_argument(
        "--type", default="residential_house",
        choices=["residential_house", "l_shaped_house", "polygon_house"],
    )
    parser.add_argument("--floors",      type=int,   default=2)
    parser.add_argument("--roof",        choices=["flat", "gable", "hip"], default="gable")
    parser.add_argument("--width",       type=float, default=10.0)
    parser.add_argument("--depth",       type=float, default=8.0)
    parser.add_argument("--pitch",       type=float, default=0.5)
    parser.add_argument("--notch-width", type=float, default=4.0)
    parser.add_argument("--notch-depth", type=float, default=3.0)
    parser.add_argument("--clay", action="store_true",
                        help="Start in clay mode instead of textured")
    args = parser.parse_args()

    roof_map = {"flat": RoofType.FLAT, "gable": RoofType.GABLE, "hip": RoofType.HIP}

    if args.type == "l_shaped_house":
        params = LShapedHouseParams(
            main_width=args.width, main_depth=args.depth,
            notch_width=args.notch_width, notch_depth=args.notch_depth,
            num_floors=args.floors,
        )
    elif args.type == "polygon_house":
        params = PolygonHouseParams(
            verts=[(0,0),(args.width,0),(args.width,args.depth),(0,args.depth)],
            num_floors=args.floors,
        )
    else:
        params = ResidentialHouseParams(
            num_floors=args.floors, roof_type=roof_map[args.roof],
            width=args.width, depth=args.depth, roof_pitch=args.pitch,
        )

    try:
        BuildingViewer(args.type, params, clay_mode=args.clay).run()
    except KeyError as exc:
        available = procbuilding.list_building_types()
        print(f"Unknown building type {exc}. Available: {available}")
        raise SystemExit(1)
