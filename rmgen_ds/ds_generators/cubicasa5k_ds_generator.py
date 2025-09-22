import json
from rmgen_ds.parameters_scene import (
    ParametersScene, ParametersSceneObject,
    ParametersCeiling, ParametersFloor,
    ParametersWall, ParametersWallOpening,
    dump_parameters_scene
)
from rmgen_ds.ds_generators.ds_generator import DSGenerator

from CubiCasa5k.floortrans.loaders.house import House

from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.draw import line, polygon
from pathlib import Path
import cv2
from typing import Literal
import numpy as np
import shapely as shp


class CubiCasa5kDSGenerator(DSGenerator):
    @staticmethod
    def get_parameters(
        house: Path | str | House,
        # tx_xy: tuple[float, float] = None,
        freq: float = 3.5e9,
        bandwidth: float = 10e6,
        wall_height: float = 3.,
        xy_scale_to_meters: float | Literal["ESTIMATE_FROM_DOORS"] = "ESTIMATE_FROM_DOORS"
    ) -> ParametersScene:
        """
        Parameters
        house:
            path to a house, an already loaded House
        freq:
            transmission frequency [Hz]
        bandwidth:
            transmission bandwidth [Hz]
        wall_height:
            distance from floor to ceiling on scenario [m]
        xy_scale_to_meters:
            how to scale xy coordinates to meters
        """
        if isinstance(house, str) or isinstance(house, Path):
            house = path2house(house)

        if xy_scale_to_meters == "ESTIMATE_FROM_DOORS":
            xy_scale_to_meters = estimate_scale_from_doors(
                house
            )
            # print("xy_scale_to_meters", xy_scale_to_meters)
            # print("1/xy_scale_to_meters", 1/xy_scale_to_meters)

        objects = house2objects(house, xy_scale_to_meters)
        tx_xy = house2candidate_txs(house, xy_scale_to_meters)
        # print(tx_xy)

        return ParametersScene(
            frequency=freq,
            bandwidth=bandwidth,
            tx_xy=tx_xy,
            wall_height=wall_height,
            objects=objects,
        )

def path2objects(
    path: str,
    scale_xy_to_meters: float,
) -> list[ParametersSceneObject]:
    house = path2house(path)
    objs = house2objects(house, scale_xy_to_meters)

    return objs

def path2house(path: Path | str) -> House:
    # orig_img_path = Path(path) / 'F1_original.png'
    img_path = Path(path) / 'F1_scaled.png'
    svg_path = Path(path) / 'model.svg'

    # TODO: don't open image, use some other lib to get only dimension
    fplan = cv2.imread(str(img_path.resolve()))
    # fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
    height, width, _ = fplan.shape

    return House(str(svg_path.resolve()), height, width)

def _clip_outside(rr, cc, height, width):
    s = np.column_stack((rr, cc))
    s = s[s[:, 0] < height]
    s = s[s[:, 1] < width]

    return s[:, 0], s[:, 1]

def house2objects(
    house: House,
    scale: float,
) -> list[ParametersSceneObject]:
    """
    Parameters
    house: House
    scale: pixel to meter conversion scale
    """
    objects = [
        ParametersCeiling(
            material="concrete",
            face_width=0.4,
        ),
        ParametersFloor(
            material="concrete",
            face_width=0.4,
        ),
    ]
    # img = np.zeros((len(house.wall_objs), house.width, house.height))
    # img = np.zeros((house.width, house.height))
    # print("(house.width, house.height)", (house.width, house.height))
    # print("[wall.X for wall in house.wall_objs]", [len(wall.X) for wall in house.wall_objs])
    # print("X", house.wall_objs[0].X)
    # print("Y", house.wall_objs[0].Y)
    all_x = np.array([x for wall in house.wall_objs for x in wall.X])
    minx, maxx = np.min(all_x), np.max(all_x)
    all_y = np.ravel([y for wall in house.wall_objs for y in wall.Y])
    miny, maxy = np.min(all_y), np.max(all_y)
    # dislocate to center scene
    dx = (maxx - minx) / 2
    dy = (maxy - miny) / 2

    for i, wall in enumerate(house.wall_objs):
        vertices = np.transpose(np.stack([wall.X - dx, wall.Y - dy])) * scale
        rr, cc = polygon(
            wall.Y,
            wall.X,
        )
        rr, cc = _clip_outside(rr, cc, house.height, house.width)
        # img[i][cc, rr] = 2 * i + 1
        # img[cc, rr] = 2 * i + 1
        # img[cc, rr] = 1
        # NOTE: min and max width are equal
        # didn't really dig deep enough to check which is true
        face_width = wall.min_width * scale / 2

        # print("wall.Y", wall.Y)
        # print("wall.X", wall.X)
        # print("wall.cc", cc)
        # print("wall.rr", rr)
        # if i in [0, 20]:
        #     # print("wall.cc", cc)
        #     # print("wall.rr", rr)
        #     fig = plt.figure()
        #     ax = fig.subplots()
        #     ax.set_axis_off()
        #     # rsh_walls
        #     im = ax.imshow(
        #         img[i]
        #     )
        #     ax.set_title("ROOM")
        #     continue

        objects.append(
            ParametersWall(
                corners_2d=vertices,
                material="concrete",
                face_width=face_width,
            )
        )

    # fig = plt.figure()
    # ax = fig.subplots()
    # ax.set_axis_off()
    # im = ax.imshow(
    #     img
    # )
    # ax.set_title("ROOM")

    # plt.show()

    for door in house.door_objs:
        vertices = np.transpose(np.stack([door.X - dx, door.Y - dy])) * scale
        # NOTE: min and max width are equal
        # didn't really dig deep enough to check which is true
        face_width = door.min_width * scale / 2

        objects.append(
            ParametersWallOpening(
                corners_2d=vertices,
                material="concrete",
                face_width=face_width,
                z_range=(0, 3.0),
                state="open"
            )
        )

    for window in house.window_objs:
        vertices = np.transpose(np.stack([window.X - dx, window.Y - dy])) * scale
        # NOTE: min and max width are equal
        # didn't really dig deep enough to check which is true
        face_width = window.min_width * scale / 2

        objects.append(
            ParametersWallOpening(
                corners_2d=vertices,
                material="glass",
                face_width=face_width,
                z_range=(1.0, 2.1),
                state="closed"
            )
        )

    return objects

def house2candidate_txs(
    house: House,
    scale: float,
) -> list[tuple[int, int]]:
    """
    Parameters
    house: House
    scale: pixel to meter conversion scale
    """
    candidates = []

    all_x = np.array([x for wall in house.wall_objs for x in wall.X])
    minx, maxx = np.min(all_x), np.max(all_x)
    all_y = np.ravel([y for wall in house.wall_objs for y in wall.Y])
    miny, maxy = np.min(all_y), np.max(all_y)
    # dislocate to center scene
    dx = (maxx - minx) / 2
    dy = (maxy - miny) / 2

    for i, room in enumerate(house.room_objs):
        xs = scale * (np.append(room.X, room.X[0]) - dx)
        ys = scale * (np.append(room.Y, room.Y[0]) - dy)
        vertices = np.transpose(np.stack([xs, ys]))
        poly = shp.Polygon(
            vertices
        )
        cent = poly.centroid
        candidates.append((cent.x, cent.y))

    return candidates

def estimate_scale_from_doors(
    house: House
) -> float:
    DOOR_WIDTH = 0.8

    if len(house.door_objs) == 0:
        raise ValueError("There are no doors in house for door based estimation")

    door_ls = [door.length for door in house.door_objs]
    l = np.mean(door_ls)

    return DOOR_WIDTH / l
            

if __name__ == "__main__":
    # house = Path("/home/artistreak/Downloads/cubicasa5k/high_quality/17")
    cubicasa_p = Path("/home/artistreak/Downloads/cubicasa5k")
    freq_GHz = 60
    INPUTS_DIR = Path(f"./datasets/concrete/{freq_GHz}GHz")
    INPUTS_DIR.mkdir(exist_ok=True, parents=True)
    ps = [
        cubicasa_p / "test.txt",
        cubicasa_p / "train.txt",
        cubicasa_p / "val.txt",
    ]
    fs = [
        fname[1:] for p in ps for fname in p.read_text().split('\n')
    ]

    def process_file(f):
        try:
            params = CubiCasa5kDSGenerator.get_parameters(
                cubicasa_p / f,
                freq=freq_GHz * 1e9,
            )
            to = INPUTS_DIR / f
            to.mkdir(exist_ok=True, parents=True)
            dump_parameters_scene(params, to / "inp.yaml")
            return f
        except Exception as e:
            return ({
                "at": f,
                "err": str(e)
            })
        # t.set_postfix(errors=len(errors),successes=len(successes))
        # (INPUTS_DIR / f"res-{os.getpid()}.json").write_text(json.dumps(
        #     {
        #         "errors": errors,
        #         "successes": successes,
        #     },
        # ))
    from tqdm.contrib.concurrent import process_map
    r = process_map(process_file, fs, max_workers=10, chunksize=10)
    errors = [x for x in r if isinstance(x, dict)]
    successes = [x for x in r if not isinstance(x, dict)]
    (INPUTS_DIR / "res.json").write_text(json.dumps(
        {
            "errors": errors,
            "successes": successes,
        },
    ))

    print("len(errors)", len(errors))
    print("len(successes)", len(successes))
    # print("(errors)", (errors))
    # house = Path("/home/artistreak/Downloads/cubicasa5k/high_quality/231")
    # params = CubiCasa5kDSGenerator.get_parameters(
    #     house,
    #     [(0., 0.)],
    # )
    # print("params", params)
    # dump_parameters_scene(params, "./scene-test.yaml")