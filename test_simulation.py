from tidy3d import *
import web
import viz

""" ==== Example simulation instance ==== """

sim = Simulation(
    mesh=Mesh(
        geometry=Cuboid(
            size=(2.0, 2.0, 2.0),
            center=(0, 0, 0)
        ),
        grid_step=(0.01, 0.01, 0.01),
        run_time=1e-12
    ),
    structures={
        "square": Structure(
            geometry=Cuboid(size=(1, 1, 1), center=(-1, 0, 0)),
            medium=Medium(permittivity=2.0),
        ),
        "box": Structure(
            geometry=Cuboid(size=(1, 1, 1), center=(0, 0, 0)),
            medium=Medium(permittivity=1.0, conductivity=3.0),
        ),
    },
    sources={
        "dipole": Source(
            geometry=Cuboid(size=(0, 0, 0), center=(0, -0.5, 0)),
            polarization=(1, 0, 1),
            source_time=Pulse(
                freq0=1e14,
                fwidth=1e12,
            ),
        )
    },
    monitors={
        "point": Monitor(
            geometry=Cuboid(size=(0, 0, 0), center=(0, 1, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        "plane": Monitor(
            geometry=Cuboid(size=(1, 1, 0), center=(0, 0, 0)),
            monitor_time=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    },
    symmetry=(0, -1, 1),
    pml_layers=(
        PMLLayer(profile="absorber", num_layers=20),
        PMLLayer(profile="stable", num_layers=30),
        PMLLayer(profile="standard"),
    ),
    shutoff=1e-6,
    courant=0.8,
    subpixel=False,
)

# example usage
if __name__ == "__main__":
    web.run(sim)
    # viz.viz_data(sim, "plane")  # vizualize

""" unit tests """

import pytest

def test_negative_sizes():

    for size in (-1, 1, 1), (1, -1, 1), (1, 1, -1):
        with pytest.raises(pydantic.ValidationError) as e_info:
            a = Cuboid(size=size, center=(0, 0, 0))

        with pytest.raises(pydantic.ValidationError) as e_info:
            m = Mesh(mesh_step=size)

def test_medium():

    with pytest.raises(pydantic.ValidationError) as e_info:
        m = Medium(permittivity=0.0)
        m = Medium(conductivity=-1.0)

def test_bounds():

    # make sure all things are shifted to this central location
    CENTER_SHIFT = (-1., 1., 100.)

    def place_box(center_offset):

        shifted_center = tuple(c + s for (c, s) in zip(center_offset, CENTER_SHIFT))

        sim = Simulation(
            mesh=Mesh(
                geometry=Cuboid(size=(1, 1, 1),
                                center=CENTER_SHIFT),
                grid_step=(0.1, 0.1, 0.1),
                run_time=1e-12
            ),
            structures={
                'box': Structure(
                    geometry=Cuboid(
                        size=(1, 1, 1),
                        center=shifted_center
                    ),
                    medium=Medium()
                )
            }
        )

    # create all permutations of squares being shifted 1, -1, or zero in all three directions
    bin_strings = [list(format(i, '03b')) for i in range(8)]
    bin_ints = [[int(b) for b in bin_string] for bin_string in bin_strings]
    bin_ints = np.array(bin_ints)
    bin_signs = 2*(bin_ints - 0.5)

    # test all cases where box is shifted +/- 1 in x,y,z and still intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = amp*sign
            place_box(tuple(center))

    # test all cases where box is shifted +/- 2 in x,y,z and no longer intersects
    for amp in bin_ints:
        for sign in bin_signs:
            center = 2 * amp * sign
            if np.sum(center) < 1e-12:
                continue
            with pytest.raises(pydantic.ValidationError) as e_info:
                place_box(tuple(center))