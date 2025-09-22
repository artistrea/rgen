import json
from rmgen_ds.parameters_scene import (
    ParametersScene, load_parameters_scene,
    ParametersWall, ParametersWallOpening
)
from sionna.rt import (
    Scene, SceneObject, ITURadioMaterial, load_scene,
    Camera, RadioMapSolver, PlanarArray, Transmitter
)
import trimesh
from pathlib import Path
import numpy as np
from shapely.geometry import box, Polygon, JOIN_STYLE
import shapely as shp
import shapely.plotting
import shapely.validation
import tempfile
import matplotlib.pyplot as plt
from skimage.draw import polygon
import cv2
from tqdm import tqdm


# def tesselate_wall_with_openings(wall, openings, wall_height):
#     """
#     wall: ParametersWall
#     openings: list of ParametersWallOpening
#     wall_height: float
#     returns: trimesh.Trimesh
#     """

#     # Wall vector and local axes
#     wx, wy = wall.from_xy
#     tx, ty = wall.to_xy
#     wall_vec = np.array([tx-wx, ty-wy, 0.])
#     wall_len = np.linalg.norm(wall_vec)
#     if wall_len == 0:
#         raise ValueError("Zero-length wall")
#     dir_along = wall_vec / wall_len
#     up = np.array([0., 0., 1.])
#     origin = np.array([wx, wy, 0.])

#     # Start with full wall rectangle in local coords
#     wall_poly = box(0, 0, wall_len, wall_height)

#     # Convert openings to wall-local 2D and subtract
#     for op in openings:
#         if not opening_on_wall(wall, op):
#             continue
#         # opening offset along wall
#         start = np.linalg.norm(np.array(op.from_xyz[:2]) - np.array([wx, wy]))
#         end = np.linalg.norm(np.array(op.to_xyz[:2]) - np.array([wx, wy]))
#         bottom = op.from_xyz[2]
#         top = op.to_xyz[2]
#         hole_poly = box(start, bottom, end, top)
#         wall_poly = wall_poly.difference(hole_poly)

#     # wall_poly is a shapely polygon (with holes)
#     verts2d, faces = trimesh.creation.triangulate_polygon(wall_poly)

#     verts3d = []
#     for v in verts2d:
#         X, Y = v
#         world = origin + dir_along * X + up * Y
#         verts3d.append(world)

#     return trimesh.Trimesh(vertices=np.array(verts3d),
#                            faces=faces,
#                            process=False)

# def opening_on_wall(wall, opening, tol=1e-6):
#     """
#     wall: ParametersWall
#     opening: ParametersWallOpening
#     tol: maximum perpendicular distance from wall
#     """
#     # Wall vector
#     wx, wy = wall.from_xy
#     tx, ty = wall.to_xy
#     wall_vec = np.array([tx - wx, ty - wy])
#     wall_len = np.linalg.norm(wall_vec)
#     if wall_len < tol:
#         return False
#     wall_dir = wall_vec / wall_len

#     # Opening center
#     ox, oy, _ = opening.from_xyz
#     cx, cy, _ = opening.to_xyz
#     op_start = np.array([ox, oy])
#     op_end = np.array([cx, cy])

#     # Vector from wall start to opening center
#     vec_to_start = op_start - np.array([wx, wy])
#     vec_to_end = op_end - np.array([wx, wy])

#     # Projection along wall
#     proj_len = np.dot(vec_to_start, wall_dir)
#     if proj_len < -tol or proj_len > wall_len + tol:
#         return False
#     # Projection along wall
#     proj_len = np.dot(vec_to_end, wall_dir)
#     if proj_len < -tol or proj_len > wall_len + tol:
#         return False

#     # Perpendicular distance to wall line
#     perp_vec = vec_to_end - proj_len * wall_dir
#     perp_dist = np.linalg.norm(perp_vec)
#     if perp_dist > tol:
#         return False

#     return True
def opening_on_wall(
    wall: ParametersWall, opening: ParametersWallOpening, eps=1e-2
) -> tuple[bool, bool]:
    wall_poly = Polygon(wall.corners_2d + (wall.corners_2d[0],))
    open_poly = Polygon(opening.corners_2d + (opening.corners_2d[0],))
    open_poly = open_poly.buffer(eps, join_style=JOIN_STYLE.mitre)

    if wall_poly.intersects(open_poly):
        on_wall = True
        diff = wall_poly.difference(open_poly)
        diff = diff.buffer(eps, join_style=JOIN_STYLE.mitre)
        if isinstance(diff, Polygon):
            # should be multipolygon, since we just divided it in 2
            print("poly")

            # fig, ax = plt.subplots()
            # shp.plotting.plot_polygon(wall_poly, ax=ax, add_points=False, facecolor="lightgray", edgecolor="black")
            # shp.plotting.plot_polygon(open_poly, ax=ax, add_points=False, facecolor="red", edgecolor="black")
            # # plt.show()

            # fig, ax = plt.subplots()
            # shp.plotting.plot_polygon(diff, ax=ax, add_points=False, facecolor="lightgray", edgecolor="black")
            # plt.show()

            has_undesired_hole = True
        elif diff.geom_type == "MultiPolygon":
            has_undesired_hole = False
            if len(diff.geoms) != 2:
                print("2 geoms")
                has_undesired_hole = True
            for pol in diff.geoms:
                if len(pol.interiors) > 0:
                    print("has interrior")
                    has_undesired_hole = True
        else:
            has_undesired_hole = False
            raise ValueError("wut")
    else:
        on_wall = False
        has_undesired_hole = False

    return on_wall, has_undesired_hole

def wall_to_meshes(wall: ParametersWall, height: float, openings: list[ParametersWallOpening]):
    # ops_poly = []
    wall_poly = trimesh.path.polygons.Polygon(wall.corners_2d + (wall.corners_2d[0],))
    for opening in openings:
        open_poly = trimesh.path.polygons.Polygon(opening.corners_2d + (opening.corners_2d[0],))
        open_poly = open_poly.buffer(1e-2, join_style=JOIN_STYLE.mitre)
        wall_poly = wall_poly.difference(open_poly)
        # ops_poly.append(
            
        # )
    if not wall_poly.is_valid:
        wall_poly = wall_poly.buffer(0)
        if not wall_poly.is_valid:
            raise ValueError(shp.validation.explain_validity(wall_poly))
    if wall_poly.geom_type == "MultiPolygon":
        return trimesh.util.concatenate([
            trimesh.creation.extrude_polygon(p, height)
            for p in wall_poly.geoms
        ])
    return trimesh.creation.extrude_polygon(wall_poly, height)

def opening_to_mesh(opening: ParametersWallOpening, eps=1e-2):
    poly = trimesh.path.polygons.Polygon(opening.corners_2d + (opening.corners_2d[0],))
    # subtraction does not work properly on overlapping planes so we add an eps
    poly = poly.buffer(eps, join_style=JOIN_STYLE.mitre)
    if not poly.is_valid:
        raise ValueError("OPAA")
    height = opening.z_range[1] - opening.z_range[0] + 2 * eps
    o_mesh = trimesh.creation.extrude_polygon(poly, height)
    o_mesh.apply_translation([0, 0, opening.z_range[0] - eps])
    return o_mesh

def update_scene_with_parameters(
    scene: Scene,
    parameters: ParametersScene,
) -> None:
    scene.frequency = parameters.frequency
    scene.bandwidth = parameters.bandwidth

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 pattern="iso",
                                 polarization="V")

    scene.rx_array = scene.tx_array

    for i, tx_xy in enumerate(parameters.tx_xy):
        tx = Transmitter(name=f"tx{i}",
                          position=[tx_xy[0], tx_xy[1], 1.5],
                          orientation=[np.pi*5/6, 0, 0],
                          power_dbm=44)
        scene.add(tx)

    walls = []
    wall_openings = []
    for obj in parameters.objects:
        if obj.type == "wall":
            walls.append(obj)
        elif obj.type == "wall_opening":
            wall_openings.append(obj)
        elif obj.type == "ceiling":
            ceiling = obj
        elif obj.type == "floor":
            floor = obj

    DIR = Path("./objs")
    DIR.mkdir(exist_ok=True)
    wall_meshes = []
    closed_opening_meshes = []

    # check all openings are accounted for
    # and none are useless
    for c, opening in enumerate(wall_openings):
        found = False
        for i in range(len(walls)):
            on_wall, undesirable = opening_on_wall(walls[i], wall_openings[c])

            if undesirable:
                raise ValueError("Subtraction causes unexpected behavior")
            if on_wall:
                found = True

        if not found:
            print("Opa, buraco onde não tem parede?")
            # raise ValueError("NOO")

    for i in range(len(walls)):
        # print("i", i)
        openings = [
            opening
                for opening in wall_openings
                if opening.state == "open" and opening_on_wall(walls[i], opening)[0]
        ]
        wall_mesh = wall_to_meshes(walls[i], parameters.wall_height, openings)
        # o_meshes = [opening_to_mesh(op) for op in openings]
        # o_meshes = [opening_to_mesh(op) for op in openings if op.state == "open"]
        # if o_meshes:
        #     combined_openings = trimesh.util.concatenate(o_meshes)
        #     # wall_meshes.append(combined_openings)
        #     wall_mesh = wall_mesh.difference(combined_openings)
        # print("wall_mesh.is_watertight", wall_mesh.is_watertight)
        # print("wall_mesh.is_volume", wall_mesh.is_volume)
        wall_mesh.fix_normals()
        if not wall_mesh.is_watertight:
            raise Exception("Something went wrong")
        wall_meshes.append(wall_mesh)

    objs_to_add = []
    maxx, minx, maxy, miny = get_parameters_scene_bound(parameters)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tmp:
        tmp_path = str(Path(tmp.name).resolve())
        vertices = np.array([
            [minx, miny, 0],
            [minx, maxy, 0],
            [maxx, maxy, 0],
            [maxx, miny, 0],
        ])

        # quad faces. trimesh automatically triangulates
        faces = np.array([
            [0, 1, 2, 3],
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(tmp_path)

        floor_obj = SceneObject(
            fname=tmp_path,
            name="floor",
            radio_material=ITURadioMaterial(
                "flor-material",
                floor.material,
                thickness=floor.face_width,
                # color=(0.1, 0.1, 0.1)
            )
        )
        objs_to_add.append(floor_obj)

        ceiling_obj = SceneObject(
            fname=tmp_path,
            name="ceiling",
            radio_material=ITURadioMaterial(
                "ceiling-material",
                ceiling.material,
                thickness=ceiling.face_width,
                # color=(0.1, 0.1, 0.1)
            )
        )
        ceiling_obj.position[2] = 3.0
        objs_to_add.append(ceiling_obj)

    for i, mesh in enumerate(wall_meshes):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tmp:
            tmp_path = str(Path(tmp.name).resolve())
            mesh.export(tmp_path)
            obj = SceneObject(
                fname=tmp_path,
                name=f"wall-{i}",
                radio_material=ITURadioMaterial(
                    f"wall-{i}-material",
                    walls[i].material,
                    thickness=walls[i].face_width,
                    # color=(0.8, 0.1, 0.1)
                )
            )
            objs_to_add.append(obj)
    scene.edit(add=objs_to_add)

def get_parameters_scene_bound(parameters:ParametersScene):
    walls = []
    for obj in parameters.objects:
        if obj.type == "wall":
            walls.append(obj)

    maxx, minx, maxy, miny = -np.inf, np.inf, -np.inf, np.inf
    for wall in walls:
        corners = np.array(wall.corners_2d)
        xs = corners[:, 0]
        ys = corners[:, 1]
        maxx = np.max([maxx, np.max(xs)])
        minx = np.min([minx, np.min(xs)])
        maxy = np.max([maxy, np.max(ys)])
        miny = np.min([miny, np.min(ys)])

    return maxx, minx, maxy, miny

def parameters_scene_image(parameters: ParametersScene, reverse_scale: float):
    walls = []
    wall_openings = []
    for obj in parameters.objects:
        if obj.type == "wall":
            walls.append(obj)
        elif obj.type == "wall_opening":
            wall_openings.append(obj)
        elif obj.type == "ceiling":
            ceiling = obj
        elif obj.type == "floor":
            floor = obj
    DIR = Path("./objs")
    DIR.mkdir(exist_ok=True)

    maxx, minx, maxy, miny = get_parameters_scene_bound(parameters)

    adj = lambda x: np.ceil(x * reverse_scale).astype(int)
    w = adj(maxx - minx) + 1
    h = adj(maxy - miny) + 1
    img = np.zeros((w, h))

    for wall in walls:
        rr, cc = pol_from_corners(wall.corners_2d, minx, miny, reverse_scale)
        img[cc, rr] = 1

    for opening in wall_openings:
        rr, cc = pol_from_corners(opening.corners_2d, minx, miny, reverse_scale)
        if opening.state == "closed":
            img[cc, rr] = 2
        else:
            img[cc, rr] = 3

    return img

def pol_from_corners(corners, minx, miny, reverse_scale, return_coords=True):
    corners = np.array(corners)
    xs = reverse_scale * (corners[:, 0] - minx)
    ys = reverse_scale * (corners[:, 1] - miny)

    pts = np.round(np.column_stack((xs, ys))).astype(np.int32)

    # bounding box around polygon
    minx_i, maxx_i = np.min(pts[:, 0]), np.max(pts[:, 0])
    miny_i, maxy_i = np.min(pts[:, 1]), np.max(pts[:, 1])

    w = maxx_i - minx_i + 1
    h = maxy_i - miny_i + 1

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - [minx_i, miny_i]
    cv2.fillPoly(mask, [shifted], 1)

    if return_coords:
        rr, cc = np.nonzero(mask)
        rr += miny_i
        cc += minx_i
        return rr, cc
    else:
        return mask, (minx_i, miny_i)

if __name__ == "__main__":
    cubicasa_p = Path("/home/artistreak/Downloads/cubicasa5k")
    freq_GHz = 60
    INPUTS_DIR = Path(f"./datasets/concrete/{freq_GHz}GHz")
    INPUTS_DIR.mkdir(exist_ok=True)
    ps = [
        cubicasa_p / "test.txt",
        cubicasa_p / "train.txt",
        cubicasa_p / "val.txt",
    ]
    fs = [
        fname[1:] for p in ps for fname in p.read_text().split('\n')
    ]
    # fs = ["high_quality_architectural/1191"]
    # fs = fs[0:1]
    errors = []
    successes = []
    non_existent = []

    def process_file(f):
        try:
            if not (INPUTS_DIR / f / "inp.yaml").exists():
                raise ValueError("input file Doesn't exist")
            params = load_parameters_scene(INPUTS_DIR / f / "inp.yaml")
            fplan_img = parameters_scene_image(params, 4)

            scene = load_scene()
            update_scene_with_parameters(
                scene,
                params
            )

            fplan_rgb = np.zeros((*fplan_img.shape, 3)) + 255
            fplan_rgb[fplan_img == 1] = [0, 0, 0]
            fplan_rgb[fplan_img == 2] = [0, 255, 0]
            # fplan_rgb[fplan_img == 3] = [0, 0, 255]

            cv2.imwrite(INPUTS_DIR / f / "fplan.png", fplan_rgb)

            rm_solver = RadioMapSolver()
            maxx, minx, maxy, miny  = get_parameters_scene_bound(params)
            maxx, minx, maxy, miny = map(float, [maxx, minx, maxy, miny])
            range_x = maxx - minx
            range_y = maxy - miny
            cent_x = maxx - range_x/2
            cent_y = maxy - range_y/2

            rm = rm_solver(scene=scene,
                           max_depth=20,
                           cell_size=[0.25,0.25],
                           size=(range_x, range_y),
                           center=(cent_x, cent_y, 1.5),
                           orientation=(0,0,0),
                           samples_per_tx=10**6,
                           specular_reflection=True,
                           diffuse_reflection=True,
                           refraction=False,
            )
            # Create new camera with different configuration
            my_cam = Camera(position=[30, -1, 20], look_at=[0, 0, 0])

            # Render scene with new camera*
            fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
            fig.savefig(INPUTS_DIR / f / "view1.png")
            my_cam = Camera(position=[-30, 1, 20], look_at=[0, 0, 0])

            # Render scene with new camera*
            fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
            fig.savefig(INPUTS_DIR / f / "view2.png")
            my_cam = Camera(position=[-30, -30, 30], look_at=[0, 0, 0])

            # Render scene with new camera*
            fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
            fig.savefig(INPUTS_DIR / f / "view3.png")
            plt.close("all")

            for tx_i in range(len(params.tx_xy)):
                g = 10 * np.log10(rm.path_gain[tx_i]).T
                ming = -127
                # print("ming", ming)
                maxg = 0
                # maxg = np.max(g)
                # print("maxg", maxg)
                g[g < ming] = ming
                g[g > maxg] = maxg
                norm_g = 255 * (g - ming) / (maxg - ming)
                # print("norm_g", norm_g)
                # walls_at = np.where(fplan_img == 1)[0]
                # norm_g[walls_at] = 0
                cv2.imwrite(INPUTS_DIR / f / f"rem-tx{tx_i}.png", norm_g[..., None])

            # plt.show()
            del scene
            # successes.append(f)
            return f
        except Exception as e:
            return ({
                "at": f,
                "err": str(e)
            })
    # with tqdm(fs) as t:
    #     for f in t:
    #         try:
    #             if not (INPUTS_DIR / f / "inp.yaml").exists():
    #                 non_existent.append(f)
    #                 continue
    #             params = load_parameters_scene(INPUTS_DIR / f / "inp.yaml")
    #             fplan_img = parameters_scene_image(params, 4)

    #             scene = load_scene()
    #             update_scene_with_parameters(
    #                 scene,
    #                 params
    #             )

    #             fplan_rgb = np.zeros((*fplan_img.shape, 3)) + 255
    #             fplan_rgb[fplan_img == 1] = [0, 0, 0]
    #             fplan_rgb[fplan_img == 2] = [0, 255, 0]
    #             # fplan_rgb[fplan_img == 3] = [0, 0, 255]

    #             cv2.imwrite(INPUTS_DIR / f / "fplan.png", fplan_rgb)

    #             rm_solver = RadioMapSolver()
    #             maxx, minx, maxy, miny  = get_parameters_scene_bound(params)
    #             maxx, minx, maxy, miny = map(float, [maxx, minx, maxy, miny])
    #             range_x = maxx - minx
    #             range_y = maxy - miny
    #             cent_x = maxx - range_x/2
    #             cent_y = maxy - range_y/2

    #             rm = rm_solver(scene=scene,
    #                            max_depth=20,
    #                            cell_size=[0.25,0.25],
    #                            size=(range_x, range_y),
    #                            center=(cent_x, cent_y, 1.5),
    #                            orientation=(0,0,0),
    #                            samples_per_tx=10**6,
    #                            specular_reflection=True,
    #                            diffuse_reflection=True,
    #                            refraction=False,
    #             )
    #             # Create new camera with different configuration
    #             my_cam = Camera(position=[30, -1, 20], look_at=[0, 0, 0])

    #             # Render scene with new camera*
    #             fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
    #             fig.savefig(INPUTS_DIR / f / "view1.png")
    #             my_cam = Camera(position=[-30, 1, 20], look_at=[0, 0, 0])

    #             # Render scene with new camera*
    #             fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
    #             fig.savefig(INPUTS_DIR / f / "view2.png")
    #             my_cam = Camera(position=[-30, -30, 30], look_at=[0, 0, 0])

    #             # Render scene with new camera*
    #             fig = scene.render(camera=my_cam, resolution=[650, 500], num_samples=8, radio_map=rm, clip_at=2.9) # Increase num_samples to increase image quality
    #             fig.savefig(INPUTS_DIR / f / "view3.png")
    #             plt.close("all")

    #             for tx_i in range(len(params.tx_xy)):
    #                 g = 10 * np.log10(rm.path_gain[tx_i]).T
    #                 ming = -127
    #                 # print("ming", ming)
    #                 maxg = 0
    #                 # maxg = np.max(g)
    #                 # print("maxg", maxg)
    #                 g[g < ming] = ming
    #                 g[g > maxg] = maxg
    #                 norm_g = 255 * (g - ming) / (maxg - ming)
    #                 # print("norm_g", norm_g)
    #                 # walls_at = np.where(fplan_img == 1)[0]
    #                 # norm_g[walls_at] = 0
    #                 cv2.imwrite(INPUTS_DIR / f / f"rem-tx{tx_i}.png", norm_g[..., None])

    #             # plt.show()
    #             del scene
    #             successes.append(f)
    #         except Exception as e:
    #             errors.append({
    #                 "at": f,
    #                 "err": str(e)
    #             })
    #         # t.set_postfix(errors=len(errors), successes=len(successes), non_existent=len(non_existent))
    from tqdm.contrib.concurrent import process_map
    r = process_map(process_file, fs, max_workers=5, chunksize=2)
    errors = [x for x in r if isinstance(x, dict)]
    successes = [x for x in r if not isinstance(x, dict)]

    print("len(errors)", len(errors))
    print("len(successes)", len(successes))
    (INPUTS_DIR / "sim-res.json").write_text(json.dumps(
        {
            "errors": errors,
            "successes": successes,
        },
    ))
