from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal, Union, Annotated, Optional
import yaml
import json
from pathlib import Path


class ParametersBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class ParametersBaseSceneObject(ParametersBase):
    material: str
    face_width: float

class ParametersWall(ParametersBaseSceneObject):
    type: Literal["wall"] = "wall"
    corners_2d: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

class ParametersWallOpening(ParametersBaseSceneObject):
    """
    For doors and windows.
    If state='closed', material will be added to the opening.
    If state='open', a hole will be created in the wall that contains it.
    """
    type: Literal["wall_opening"] = "wall_opening"
    corners_2d: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    z_range: tuple[float, float]
    state: Literal["open", "closed"]

class ParametersCeiling(ParametersBaseSceneObject):
    type: Literal["ceiling"] = "ceiling"

class ParametersFloor(ParametersBaseSceneObject):
    type: Literal["floor"] = "floor"

ParametersSceneObject = Annotated[
    Union[ParametersWall, ParametersCeiling, ParametersFloor, ParametersWallOpening],
    Field(discriminator="type")
]

class ParametersScene(ParametersBase):
    frequency: float
    bandwidth: float
    tx_xy: list[tuple[float, float]]

    wall_height: float
    objects: list[ParametersSceneObject]
    
    @field_validator("objects")
    @classmethod
    def must_contain_ceiling_and_floor(
        cls, objs: list[ParametersSceneObject]
    ) -> list[ParametersSceneObject]:
        ceiling, floor = False, False

        for obj in objs:
            if obj.type == "floor":
                if floor:
                    raise ValueError("A scene may only have a single floor")
                floor = True

            if obj.type == "ceiling":
                if ceiling:
                    raise ValueError("A scene may only have a single ceiling")
                ceiling = True

        if not ceiling or not floor:
            raise ValueError("A scene must contain ceiling and floor")

        return objs

def create_parameters_scene(
    d: dict
) -> ParametersScene:
    return ParametersScene.model_validate(d)


def load_parameters_scene(path: str | Path):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return create_parameters_scene(data)

def dump_parameters_scene(
    params: ParametersScene,
    path: str | Path,
) -> None:
    path = Path(path)
    data = params.model_dump()

    if path.suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(data, indent=2, sort_keys=False)
    elif path.suffix == ".json":
        text = json.dumps(data, indent=2)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    path.write_text(text)

if __name__ == "__main__":
    params = create_parameters_scene({
        "frequency": 3.5e9,
        "bandwidth": 5e6,

        "tx_xy": [(0, 0)],

        "wall_height": 3.0,
        "objects": [
            {
                "type": "wall",
                "from_xy":(-10, -10),
                "to_xy":(10, 10),
                # "avaaa": 3,
                "material":"itu-concrete",
                "width":0.2,
            },
            {
                "type": "ceiling",
                "material":"itu-concrete",
                "width":0.2,
            },
            {
                "type": "floor",
                "material":"itu-concrete",
                "width":0.2,
            },
        ]
    })

    print("params", params)

    import tempfile

    # create a temporary file (will be deleted when closed)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".yaml") as tmp:
        tmp_path = Path(tmp.name)

        # dump and load from file
        dump_parameters_scene(
            params, tmp_path
        )
        tmp_params = load_parameters_scene(
            tmp_path
        )

        print("tmp_path.read_text()", tmp_path.read_text())
        print("tmp_params", tmp_params)
