"""PLY to USD/USDZ converter with vertex color support."""

from pathlib import Path

from isaacnav.conversion.base import BaseConverter, ConversionResult


class PlyToUsdConverter(BaseConverter):
    """Convert PLY meshes to USD/USDZ with vertex colors."""

    def __init__(self, config: dict):
        self.config = config
        self.scale = config.get("scale", 1.0)
        self.up_axis = config.get("up_axis", "Z")
        self.alpha_shape_alpha = config.get("alpha_shape_alpha", 0.03)

    def supported_input_formats(self) -> list[str]:
        return [".ply"]

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        scale: float = None,
    ) -> ConversionResult:
        import open3d as o3d
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        scale = scale or self.scale
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If output is .usdz, first create .usd then package
        is_usdz = output_path.suffix == ".usdz"
        if is_usdz:
            usd_path = output_path.with_suffix(".usd")
        else:
            usd_path = output_path

        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if not mesh.has_triangles():
            print(f"Warning: {input_path} has no triangles, creating mesh from point cloud")
            pcd = o3d.io.read_point_cloud(str(input_path))
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=self.alpha_shape_alpha
            )

        mesh.compute_vertex_normals()

        vertices = mesh.vertices
        triangles = mesh.triangles
        normals = mesh.vertex_normals
        colors = mesh.vertex_colors if mesh.has_vertex_colors() else None

        # Create USD stage
        stage = Usd.Stage.CreateNew(str(usd_path))
        stage.SetMetadata("metersPerUnit", 1.0)
        stage.SetMetadata("upAxis", self.up_axis)

        usd_mesh = UsdGeom.Mesh.Define(stage, "/World/mesh")

        points = [
            Gf.Vec3f(v[0] * scale, v[1] * scale, v[2] * scale) for v in vertices
        ]
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

        face_counts = [3] * len(triangles)
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))

        indices = []
        for tri in triangles:
            indices.extend([int(tri[0]), int(tri[1]), int(tri[2])])
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))

        if len(normals) > 0:
            normal_data = [Gf.Vec3f(n[0], n[1], n[2]) for n in normals]
            usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normal_data))
            usd_mesh.SetNormalsInterpolation("vertex")

        if colors is not None and len(colors) > 0:
            color_data = [Gf.Vec3f(c[0], c[1], c[2]) for c in colors]
            usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray(color_data))
            usd_mesh.GetDisplayColorPrimvar().SetInterpolation("vertex")

        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
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
            has_texture=False,
            texture_path=None,
        )
