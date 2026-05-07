"""GLB to USD/USDZ converter with texture support."""

from pathlib import Path

import numpy as np

from isaacnav.conversion.base import BaseConverter, ConversionResult


class GlbToUsdConverter(BaseConverter):
    """Convert GLB meshes to USD/USDZ with textures."""

    def __init__(self, config: dict):
        self.config = config
        self.scale = config.get("scale", 1.0)
        self.up_axis = config.get("up_axis", "Z")
        self.output_usdz = config.get("output_usdz", True)

    def supported_input_formats(self) -> list[str]:
        return [".glb", ".gltf"]

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        scale: float = None,
    ) -> ConversionResult:
        import trimesh
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

        scale = scale or self.scale
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output is .usdz, we first create a .usd then package it
        is_usdz = output_path.suffix == ".usdz"
        if is_usdz:
            usd_path = output_path.with_suffix(".usd")
        else:
            usd_path = output_path

        # Load GLB
        scene = trimesh.load(str(input_path))
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
            if not meshes:
                raise ValueError(f"No meshes found in {input_path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        # Create USD stage
        stage = Usd.Stage.CreateNew(str(usd_path))
        stage.SetMetadata("metersPerUnit", 1.0)
        stage.SetMetadata("upAxis", self.up_axis)

        world_prim = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_prim.GetPrim())

        usd_mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")

        # Set vertices
        vertices = mesh.vertices * scale
        points = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        # Set faces
        faces = mesh.faces
        face_counts = [3] * len(faces)
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
        indices = faces.flatten().tolist()
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))

        # Set normals
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
            normals = [
                Gf.Vec3f(float(n[0]), float(n[1]), float(n[2]))
                for n in mesh.vertex_normals
            ]
            usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normals))
            usd_mesh.SetNormalsInterpolation("vertex")

        # Handle texture/material
        texture_saved = False
        texture_path = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
            material = mesh.visual.material

            if hasattr(material, "image") and material.image is not None:
                texture_dir = usd_path.parent / "textures"
                texture_dir.mkdir(exist_ok=True)
                texture_filename = usd_path.stem + "_diffuse.png"
                texture_path = texture_dir / texture_filename

                material.image.save(str(texture_path))
                print(f"Saved texture: {texture_path}")

                # Create USD material with texture
                mat_path = "/World/Materials/DiffuseMaterial"
                usd_material = UsdShade.Material.Define(stage, mat_path)

                shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")

                tex_reader = UsdShade.Shader.Define(stage, mat_path + "/DiffuseTexture")
                tex_reader.CreateIdAttr("UsdUVTexture")
                tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                    f"./textures/{texture_filename}"
                )
                tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

                shader.CreateInput(
                    "diffuseColor", Sdf.ValueTypeNames.Color3f
                ).ConnectToSource(tex_reader.ConnectableAPI(), "rgb")
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

                usd_material.CreateSurfaceOutput().ConnectToSource(
                    shader.ConnectableAPI(), "surface"
                )

                # Set UVs
                if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
                    uvs = mesh.visual.uv
                    uv_data = [Gf.Vec2f(float(uv[0]), float(uv[1])) for uv in uvs]
                    uv_primvar = usd_mesh.CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                    )
                    uv_primvar.Set(Vt.Vec2fArray(uv_data))

                UsdShade.MaterialBindingAPI(usd_mesh).Bind(usd_material)
                texture_saved = True

        # Handle vertex colors (common output from SAM-3D-objects)
        if not texture_saved and hasattr(mesh, "visual"):
            colors = None
            if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors
            elif hasattr(mesh.visual, "to_color") and callable(mesh.visual.to_color):
                try:
                    color_visual = mesh.visual.to_color()
                    if hasattr(color_visual, "vertex_colors"):
                        colors = color_visual.vertex_colors
                except Exception:
                    pass

            if colors is not None and len(colors) > 0:
                colors = np.array(colors)
                if colors.max() > 1.0:
                    colors = colors.astype(float) / 255.0
                color_data = [
                    Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors[:, :3]
                ]
                usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray(color_data))
                usd_mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")
                print(f"Applied vertex colors ({len(colors)} vertices)")

        stage.GetRootLayer().Save()
        print(f"Converted: {input_path} -> {usd_path}")

        # Package as USDZ if requested
        final_path = usd_path
        if is_usdz:
            from pxr import UsdUtils

            success = UsdUtils.CreateNewUsdzPackage(
                Sdf.AssetPath(str(usd_path)), str(output_path)
            )
            if success:
                final_path = output_path
                print(f"Packaged USDZ: {output_path}")
            else:
                print(f"Warning: USDZ packaging failed, keeping USD: {usd_path}")
                final_path = usd_path

        return ConversionResult(
            output_path=final_path,
            format=final_path.suffix.lstrip("."),
            has_texture=texture_saved,
            texture_path=texture_path,
        )
