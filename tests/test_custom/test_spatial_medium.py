
import tidy3d as td
import custom.spatial_medium as sm
import numpy as np

def test_box():
    """Test the box class."""
    box1 = td.Box(center=(0,0,0), size=(1, 1, 1))
    box2 = td.Box(center=(0,0,0), size=(1, 1, 1))
    box3 = td.Box(center=(0,0,0), size=(2, 2, 2))
    box4 = td.Box(center=(1,1,1), size=(2, 2, 2))
    assert box1.intersects(box2)
    assert box1.intersects(box3)
    assert box3.intersects(box4)
    assert box1.intersects(box4)

def test_cell_box():
    """Test the cellbox class."""
    cell = sm.CellBox(center=(0,0,0), size=(1, 1, 1))
    assert cell.volume() == 1
    sub_cells = cell.split_evenly()
    assert len(sub_cells) == 8
    assert sub_cells[0].size == sub_cells[1].size
    sub1 = sub_cells[0]
    assert sub1.center == (-0.25, -0.25, -0.25)
    assert sub1.size == (0.5, 0.5, 0.5)
    vertices = cell.vertices()
    assert vertices.shape == (8, 3)
    assert all(vertices[0] == (-0.5, -0.5,-0.5))
    assert all(vertices[-1] == (0.5, 0.5, 0.5))

def test_cell_grid():
    """Test the cellgrid class."""
    x = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 3)
    z = np.linspace(-1, 1, 5)
    coords = td.Coords(x=x, y=y, z=z)
    grid = sm.CellGrid(boundaries=coords)
    assert grid.n_cells() == 8
    # cell index 0-7
    assert grid.encode_cell(0, 0, 0) == 0
    assert grid.encode_cell(0, 1, 1) == 3
    assert grid.decode_cell(3) == (0, 1, 1)

    adj = grid.adjacent_cell_inds(0, 0, 0)
    assert adj == [0]
    adj = grid.adjacent_cell_inds(1,2,4)
    assert adj == [7]
    adj = grid.adjacent_cell_inds(0, 1, 1)
    assert sorted(adj) == [0, 1, 2, 3]

    cell = grid.cell_at(0, 1, 1)
    assert cell.center == (0, 0.5, -0.25)
    assert cell.size == (2, 1, 0.5)

def test_cell_factor():
    cell_box = sm.CellBox(center=(0,0,0), size=(0.5, 0.5, 0.5))
    geometry1 = td.Box(center=(0,0,0), size=(1, 1, 1))
    f1 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry1)
    assert f1  == 1
    geometry2 = td.Box(center=(0,0,0), size=(0.5, 0.5, 0.5))
    f2 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry2)
    assert f2 == 1
    geometry3 = td.Box(center=(1,0,0), size=(0.5, 0.5, 0.5))
    f3 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry3)
    assert f3 == 0
    geometry4 = td.Box(center=(0.25,0,0), size=(0.5, 0.5, 0.5))
    f4 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry4)
    assert f4 == 0.5
    geometry5 = td.Box(center=(0.25,0.25,0.25), size=(0.5, 0.5, 0.5))
    f5 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry5)
    assert f5 == 0.125
    geometry6 = td.Box(center=(0, 0, 0), size=(0.25, 0.25, 0.25))
    f6 = sm.SpatialMediumCreator.calc_cell_factor(cell=cell_box, geometry=geometry6)
    assert f6 == 0.125

def test_spatial_medium():
    """Test the spatial medium class."""
    # size of simulation domain
    Lx, Ly, Lz = 2,3,2
    dl = 0.0666
    # waveguide information
    wg_width = 1.5
    wg_height = 1.0
    wg_permittivity = 4.0
    box = td.Box(center=(0,0,0), size=(Lx, Ly, Lz))
    waveguide = td.Structure(
        geometry=td.Box(size=(wg_width, td.inf, wg_height)),
        medium=td.Medium(permittivity=wg_permittivity)
    )
    creator = sm.SpatialMediumCreator(box=box, dl=dl, wave_guide=waveguide)
    medium = creator.create()
    assert medium.permittivity.values.shape == (int(Lx / dl) + 1, int(Ly / dl) + 1, int(Lz / dl) + 1)




