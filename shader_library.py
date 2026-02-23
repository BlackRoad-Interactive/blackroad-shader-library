"""
BlackRoad Shader Library
GLSL shader collection and management system.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ShaderType(str, Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"


class BlendMode(str, Enum):
    ADDITIVE = "additive"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    NORMAL = "normal"


class EffectType(str, Enum):
    BLOOM = "bloom"
    BLUR = "blur"
    VIGNETTE = "vignette"
    CHROMATIC_ABERRATION = "chromatic_aberration"
    SCANLINES = "scanlines"
    NOISE = "noise"
    WARP = "warp"
    PIXELATE = "pixelate"
    EDGE_DETECT = "edge_detect"
    COLOR_GRADE = "color_grade"


# ---------------------------------------------------------------------------
# Shader dataclass
# ---------------------------------------------------------------------------

@dataclass
class Shader:
    """GLSL shader definition."""
    id: str
    name: str
    type: ShaderType
    glsl_source: str
    uniforms: Dict[str, str] = field(default_factory=dict)   # name → type
    description: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "300 es"
    created_at: float = field(default_factory=time.time)

    @property
    def line_count(self) -> int:
        return self.glsl_source.count("\n") + 1

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.glsl_source.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "uniforms": self.uniforms,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "lines": self.line_count,
            "hash": self.source_hash,
        }


# ---------------------------------------------------------------------------
# GLSL validation helpers
# ---------------------------------------------------------------------------

GLSL_BUILTINS = {
    "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
    "float", "int", "uint", "bool", "sampler2D",
    "sin", "cos", "tan", "abs", "sqrt", "pow", "exp", "log",
    "mix", "clamp", "smoothstep", "step", "length", "normalize",
    "dot", "cross", "reflect", "refract", "texture", "fract",
    "floor", "ceil", "mod", "max", "min", "sign", "distance",
}

GLSL_KEYWORDS = {
    "void", "return", "if", "else", "for", "while", "break", "continue",
    "in", "out", "inout", "uniform", "varying", "attribute", "const",
    "precision", "highp", "mediump", "lowp", "layout", "main",
}


def validate_glsl(source: str) -> dict:
    """Basic GLSL syntax validation."""
    errors = []
    warnings = []

    # Must have main()
    if "void main()" not in source and "void main (" not in source:
        errors.append("Missing void main() entry point")

    # Check balanced braces
    depth = 0
    for ch in source:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            errors.append("Unmatched closing brace '}'")
            break
    if depth != 0:
        errors.append(f"Unbalanced braces (depth={depth})")

    # Check for semicolons after statements (rough heuristic)
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        if stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(","):
            continue
        if stripped.endswith(";"):
            continue
        # Ignore blank continuations or function heads
        if re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_ ]*\(", stripped) and "{" in stripped:
            continue
        # Warn on suspicious lines
        if any(kw in stripped for kw in ["vec", "float", "int", "sampler"]):
            if not stripped.endswith(";") and "{" not in stripped and "}" not in stripped:
                warnings.append(f"Line {i}: possible missing semicolon")

    # Check for #version
    has_version = any(line.strip().startswith("#version") for line in lines)
    if not has_version:
        warnings.append("No #version directive found; consider adding #version 300 es")

    # Check for deprecated gl_FragColor (ES 3.0)
    if "gl_FragColor" in source:
        warnings.append("gl_FragColor is deprecated in GLSL ES 3.0; use out vec4 fragColor")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "lines": len(lines),
    }


# ---------------------------------------------------------------------------
# GLSL template generators
# ---------------------------------------------------------------------------

def _make_vertex_passthrough() -> str:
    return """\
#version 300 es
precision highp float;

in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;

void main() {
    v_texCoord = a_texCoord;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""


def generate_shader(effect_type: EffectType) -> str:
    """Generate template GLSL fragment shader source for a given effect."""

    if effect_type == EffectType.BLOOM:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_threshold;
uniform float u_intensity;
uniform vec2 u_resolution;
in vec2 v_texCoord;
out vec4 fragColor;

vec4 sampleBlur(sampler2D tex, vec2 uv, vec2 texel, float radius) {
    vec4 color = vec4(0.0);
    float total = 0.0;
    for (float x = -radius; x <= radius; x++) {
        for (float y = -radius; y <= radius; y++) {
            float w = exp(-(x*x + y*y) / (2.0 * radius * radius));
            color += texture(tex, uv + vec2(x, y) * texel) * w;
            total += w;
        }
    }
    return color / total;
}

void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 base = texture(u_texture, v_texCoord);
    float brightness = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec4 bloom = vec4(0.0);
    if (brightness > u_threshold) {
        bloom = sampleBlur(u_texture, v_texCoord, texel, 4.0) * u_intensity;
    }
    fragColor = clamp(base + bloom, 0.0, 1.0);
}
"""

    elif effect_type == EffectType.BLUR:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform float u_radius;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec2 texel = 1.0 / u_resolution;
    vec4 color = vec4(0.0);
    float total = 0.0;
    int r = int(u_radius);
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            float dist = float(x*x + y*y);
            float w = exp(-dist / (2.0 * u_radius * u_radius));
            color += texture(u_texture, v_texCoord + vec2(float(x), float(y)) * texel) * w;
            total += w;
        }
    }
    fragColor = color / total;
}
"""

    elif effect_type == EffectType.VIGNETTE:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_strength;
uniform float u_softness;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_texCoord);
    vec2 uv = v_texCoord - 0.5;
    float dist = length(uv);
    float vignette = smoothstep(u_strength, u_strength - u_softness, dist);
    fragColor = vec4(color.rgb * vignette, color.a);
}
"""

    elif effect_type == EffectType.CHROMATIC_ABERRATION:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_amount;
uniform vec2 u_resolution;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec2 offset = (v_texCoord - 0.5) * u_amount / u_resolution;
    float r = texture(u_texture, v_texCoord + offset).r;
    float g = texture(u_texture, v_texCoord).g;
    float b = texture(u_texture, v_texCoord - offset).b;
    float a = texture(u_texture, v_texCoord).a;
    fragColor = vec4(r, g, b, a);
}
"""

    elif effect_type == EffectType.SCANLINES:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_scanline_count;
uniform float u_intensity;
uniform float u_time;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(u_texture, v_texCoord);
    float scanline = sin(v_texCoord.y * u_scanline_count + u_time) * 0.5 + 0.5;
    color.rgb *= mix(1.0, scanline, u_intensity);
    fragColor = color;
}
"""

    elif effect_type == EffectType.NOISE:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_amount;
uniform float u_time;
in vec2 v_texCoord;
out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec4 color = texture(u_texture, v_texCoord);
    float n = noise(v_texCoord * 500.0 + u_time) * u_amount;
    color.rgb += n - u_amount * 0.5;
    fragColor = clamp(color, 0.0, 1.0);
}
"""

    elif effect_type == EffectType.WARP:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_strength;
uniform float u_frequency;
uniform float u_time;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    float wave_x = sin(v_texCoord.y * u_frequency + u_time) * u_strength;
    float wave_y = cos(v_texCoord.x * u_frequency + u_time) * u_strength;
    vec2 warped = v_texCoord + vec2(wave_x, wave_y);
    fragColor = texture(u_texture, warped);
}
"""

    elif effect_type == EffectType.PIXELATE:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_pixel_size;
uniform vec2 u_resolution;
in vec2 v_texCoord;
out vec4 fragColor;

void main() {
    vec2 size = u_pixel_size / u_resolution;
    vec2 uv = floor(v_texCoord / size) * size + size * 0.5;
    fragColor = texture(u_texture, uv);
}
"""

    elif effect_type == EffectType.EDGE_DETECT:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform float u_threshold;
in vec2 v_texCoord;
out vec4 fragColor;

float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 texel = 1.0 / u_resolution;
    float tl = luminance(texture(u_texture, v_texCoord + vec2(-texel.x,  texel.y)).rgb);
    float t  = luminance(texture(u_texture, v_texCoord + vec2( 0.0,      texel.y)).rgb);
    float tr = luminance(texture(u_texture, v_texCoord + vec2( texel.x,  texel.y)).rgb);
    float l  = luminance(texture(u_texture, v_texCoord + vec2(-texel.x,  0.0    )).rgb);
    float r  = luminance(texture(u_texture, v_texCoord + vec2( texel.x,  0.0    )).rgb);
    float bl = luminance(texture(u_texture, v_texCoord + vec2(-texel.x, -texel.y)).rgb);
    float b  = luminance(texture(u_texture, v_texCoord + vec2( 0.0,     -texel.y)).rgb);
    float br = luminance(texture(u_texture, v_texCoord + vec2( texel.x, -texel.y)).rgb);
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    float edge = length(vec2(gx, gy));
    float e = step(u_threshold, edge);
    fragColor = vec4(vec3(e), 1.0);
}
"""

    elif effect_type == EffectType.COLOR_GRADE:
        return """\
#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform vec3 u_lift;
uniform vec3 u_gamma;
uniform vec3 u_gain;
uniform float u_saturation;
in vec2 v_texCoord;
out vec4 fragColor;

vec3 colorGrade(vec3 color) {
    color = u_lift + color * u_gain;
    color = pow(max(color, 0.0), 1.0 / max(u_gamma, vec3(0.001)));
    return color;
}

void main() {
    vec4 src = texture(u_texture, v_texCoord);
    vec3 graded = colorGrade(src.rgb);
    float lum = dot(graded, vec3(0.2126, 0.7152, 0.0722));
    graded = mix(vec3(lum), graded, u_saturation);
    fragColor = vec4(clamp(graded, 0.0, 1.0), src.a);
}
"""
    else:
        return f"// Unknown effect type: {effect_type}\nvoid main() {{}}\n"


# ---------------------------------------------------------------------------
# Shader Library
# ---------------------------------------------------------------------------

class ShaderLibrary:
    """Collection and management of GLSL shaders."""

    def __init__(self):
        self._shaders: Dict[str, Shader] = {}
        self._counter: int = 0
        self._load_defaults()

    def _make_id(self, name: str) -> str:
        self._counter += 1
        slug = re.sub(r"[^a-z0-9_]", "_", name.lower())
        return f"shader_{slug}_{self._counter:04d}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_shader(
        self,
        name: str,
        shader_type: ShaderType,
        source: str,
        uniforms: Optional[Dict[str, str]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Shader:
        """Add a shader to the library."""
        sid = self._make_id(name)
        shader = Shader(
            id=sid,
            name=name,
            type=shader_type,
            glsl_source=source,
            uniforms=uniforms or {},
            description=description,
            tags=tags or [],
        )
        self._shaders[sid] = shader
        return shader

    def get_shader(self, id_or_name: str) -> Optional[Shader]:
        """Retrieve a shader by ID or name."""
        if id_or_name in self._shaders:
            return self._shaders[id_or_name]
        for s in self._shaders.values():
            if s.name == id_or_name:
                return s
        return None

    def remove_shader(self, id_or_name: str) -> bool:
        shader = self.get_shader(id_or_name)
        if shader:
            del self._shaders[shader.id]
            return True
        return False

    def update_shader(self, id_or_name: str, source: str) -> Optional[Shader]:
        shader = self.get_shader(id_or_name)
        if shader:
            shader.glsl_source = source
        return shader

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_shaders(self, type_filter: Optional[ShaderType] = None) -> List[Shader]:
        """List shaders, optionally filtered by type."""
        shaders = list(self._shaders.values())
        if type_filter:
            shaders = [s for s in shaders if s.type == type_filter]
        return sorted(shaders, key=lambda s: s.name)

    def search(self, query: str) -> List[Shader]:
        """Search shaders by name, description, or tags."""
        q = query.lower()
        results = []
        for s in self._shaders.values():
            if (q in s.name.lower() or
                    q in s.description.lower() or
                    any(q in tag.lower() for tag in s.tags)):
                results.append(s)
        return results

    def list_tags(self) -> List[str]:
        tags = set()
        for s in self._shaders.values():
            tags.update(s.tags)
        return sorted(tags)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_glsl(self, source: str) -> dict:
        return validate_glsl(source)

    def validate_shader(self, id_or_name: str) -> dict:
        shader = self.get_shader(id_or_name)
        if not shader:
            return {"valid": False, "errors": ["Shader not found"], "warnings": []}
        return validate_glsl(shader.glsl_source)

    def validate_all(self) -> Dict[str, dict]:
        return {sid: validate_glsl(s.glsl_source) for sid, s in self._shaders.items()}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_shader(self, effect_type: EffectType) -> Shader:
        """Generate and register a shader for the given effect type."""
        source = generate_shader(effect_type)
        uniforms = self._extract_uniforms(source)
        shader = self.add_shader(
            name=f"{effect_type.value}_generated",
            shader_type=ShaderType.FRAGMENT,
            source=source,
            uniforms=uniforms,
            description=f"Auto-generated {effect_type.value} effect shader",
            tags=["generated", effect_type.value, "postprocess"],
        )
        return shader

    def generate_vertex_passthrough(self) -> Shader:
        source = _make_vertex_passthrough()
        return self.add_shader(
            name="vertex_passthrough",
            shader_type=ShaderType.VERTEX,
            source=source,
            uniforms={"a_position": "vec2", "a_texCoord": "vec2"},
            description="Standard pass-through vertex shader",
            tags=["utility", "vertex"],
        )

    @staticmethod
    def _extract_uniforms(source: str) -> Dict[str, str]:
        """Parse uniform declarations from GLSL source."""
        uniforms = {}
        for match in re.finditer(r"uniform\s+(\w+)\s+(\w+)\s*;", source):
            utype, uname = match.group(1), match.group(2)
            uniforms[uname] = utype
        return uniforms

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, shader_ids: List[str], blend_mode: BlendMode = BlendMode.NORMAL) -> Shader:
        """Compose multiple fragment shaders into a single chained shader."""
        sources = []
        for sid in shader_ids:
            s = self.get_shader(sid)
            if s and s.type == ShaderType.FRAGMENT:
                sources.append(f"// === {s.name} ===\n{s.glsl_source}")

        blend_comment = {
            BlendMode.ADDITIVE: "fragColor += pass;",
            BlendMode.MULTIPLY: "fragColor *= pass;",
            BlendMode.SCREEN:   "fragColor = 1.0 - (1.0 - fragColor) * (1.0 - pass);",
            BlendMode.OVERLAY:  "fragColor = mix(2.0*fragColor*pass, 1.0-2.0*(1.0-fragColor)*(1.0-pass), step(0.5, fragColor.r));",
            BlendMode.NORMAL:   "fragColor = pass;",
        }.get(blend_mode, "fragColor = pass;")

        composed_source = (
            "#version 300 es\nprecision highp float;\n"
            "uniform sampler2D u_texture;\nin vec2 v_texCoord;\nout vec4 fragColor;\n\n"
            "// Composed from: " + ", ".join(shader_ids) + "\n"
            "// Blend mode: " + blend_mode.value + "\n\n"
            "void main() {\n"
            "    fragColor = texture(u_texture, v_texCoord);\n"
            "    vec4 pass = fragColor;\n"
            "    " + blend_comment + "\n"
            "}\n"
        )

        return self.add_shader(
            name=f"composed_{'_'.join(shader_ids[:3])}",
            shader_type=ShaderType.FRAGMENT,
            source=composed_source,
            description=f"Composition of {len(shader_ids)} shaders with {blend_mode.value} blending",
            tags=["composed"],
        )

    # ------------------------------------------------------------------
    # Export/Import
    # ------------------------------------------------------------------

    def export_bundle(self, ids: Optional[List[str]] = None) -> str:
        """Export shaders as a JSON bundle."""
        if ids is None:
            ids = list(self._shaders.keys())
        bundle = {
            "version": "1.0",
            "exported_at": time.time(),
            "shaders": [
                {**s.to_dict(), "glsl_source": s.glsl_source}
                for sid in ids
                if (s := self._shaders.get(sid))
            ],
        }
        return json.dumps(bundle, indent=2)

    def import_bundle(self, json_str: str) -> int:
        """Import shaders from a JSON bundle. Returns count imported."""
        bundle = json.loads(json_str)
        imported = 0
        for entry in bundle.get("shaders", []):
            shader = Shader(
                id=entry.get("id", self._make_id(entry["name"])),
                name=entry["name"],
                type=ShaderType(entry["type"]),
                glsl_source=entry.get("glsl_source", ""),
                uniforms=entry.get("uniforms", {}),
                description=entry.get("description", ""),
                tags=entry.get("tags", []),
            )
            self._shaders[shader.id] = shader
            imported += 1
        return imported

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def _load_defaults(self):
        """Pre-load a set of default utility shaders."""
        self.generate_vertex_passthrough()
        for effect in [EffectType.VIGNETTE, EffectType.BLUR, EffectType.NOISE]:
            self.generate_shader(effect)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        shaders = list(self._shaders.values())
        by_type: Dict[str, int] = {}
        for s in shaders:
            by_type[s.type.value] = by_type.get(s.type.value, 0) + 1
        total_lines = sum(s.line_count for s in shaders)
        return {
            "total": len(shaders),
            "by_type": by_type,
            "total_lines": total_lines,
            "tags": self.list_tags(),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== BlackRoad Shader Library Demo ===")
    lib = ShaderLibrary()

    print(f"\nDefault shaders loaded: {lib.stats()}")

    # Generate all effects
    print("\n[Generating effects...]")
    for effect in EffectType:
        s = lib.generate_shader(effect)
        print(f"  {effect.value}: {s.id} ({s.line_count} lines)")

    # Validate
    print("\n[Validating all shaders...]")
    results = lib.validate_all()
    valid = sum(1 for r in results.values() if r["valid"])
    print(f"  Valid: {valid}/{len(results)}")

    for sid, result in results.items():
        if result["warnings"]:
            name = lib._shaders[sid].name
            print(f"  Warnings for {name}: {result['warnings'][:1]}")

    # Search
    print("\n[Search: 'blur']")
    found = lib.search("blur")
    for s in found:
        print(f"  {s.name}: {s.description}")

    # Compose
    blur_id = lib.search("blur")[0].id if lib.search("blur") else None
    vignette_id = lib.search("vignette")[0].id if lib.search("vignette") else None
    if blur_id and vignette_id:
        composed = lib.compose([blur_id, vignette_id], BlendMode.NORMAL)
        print(f"\n[Composed shader]: {composed.name} ({composed.line_count} lines)")

    # Export bundle
    print("\n[Export bundle (first 3 shaders)]")
    ids = [list(lib._shaders.keys())[i] for i in range(min(3, len(lib._shaders)))]
    bundle_json = lib.export_bundle(ids)
    bundle = json.loads(bundle_json)
    print(f"  Bundle contains {len(bundle['shaders'])} shaders")

    # Final stats
    print(f"\nFinal stats: {lib.stats()}")


if __name__ == "__main__":
    demo()
