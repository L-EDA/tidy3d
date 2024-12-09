import queue

import numpy as np

import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray

ITERATIONS = 5

class CellBox(td.Box):
    def split(self, nx: int, ny: int, nz: int):
        sx, sy, sz = self.size # type: ignore
        dx = sx / nx
        dy = sy / ny
        dz = sz / nz
        boxes = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = self.center[0] - sx / 2 + (i + 0.5) * dx
                    y = self.center[1] - sy / 2 + (j + 0.5) * dy
                    z = self.center[2] - sz / 2 + (k + 0.5) * dz
                    boxes.append(CellBox(center=(x, y, z), size=(dx, dy, dz)))
        return boxes

    def split_evenly(self):
        return self.split(2, 2, 2)

    def volume(self):
        return super().volume()

    def vertices(self)-> np.ndarray:
        (x0, y0, z0), (x1, y1, z1) = self.bounds
        return np.array([
            (x0, y0, z0),
            (x0, y0, z1),
            (x0, y1, z0),
            (x0, y1, z1),
            (x1, y0, z0),
            (x1, y0, z1),
            (x1, y1, z0),
            (x1, y1, z1),
        ])

class CellGrid(td.Grid):
    """Defines a grid for the simulation domain."""

    def cell_at(self, x: int, y: int, z: int) -> CellBox:
        """Returns the cell at the given coordinates."""
        x = int(x)
        y = int(y)
        z = int(z)
        if x < 0 or x > self.num_cells[0]:
            raise ValueError(f"x={x} is out of bounds")
        if y < 0 or y > self.num_cells[1]:
            raise ValueError(f"y={y} is out of bounds")
        if z < 0 or z > self.num_cells[2]:
            raise ValueError(f"z={z} is out of bounds")
        lx = self.boundaries.x[x + 1] - self.boundaries.x[x]
        ly = self.boundaries.y[y + 1] - self.boundaries.y[y]
        lz = self.boundaries.z[z + 1] - self.boundaries.z[z]
        cx = self.boundaries.x[x] + lx / 2
        cy = self.boundaries.y[y] + ly / 2
        cz = self.boundaries.z[z] + lz / 2
        return CellBox(center=(cx, cy, cz), size=(lx, ly, lz))

    def n_cells(self):
        return np.prod(self.num_cells)

    def decode_cell(self, cell: int):
        if cell < 0 or cell >= self.n_cells():
            raise ValueError(f"cell={cell} is out of bounds")

        nx, ny, _ = self.num_cells
        x = cell % nx
        y = (cell // nx) % ny
        z = cell // (nx * ny)
        return x, y, z

    def encode_cell(self, x: int, y: int, z: int):
        nx, ny, _ = self.num_cells
        return x + nx * (y + ny * z)

    def adjacent_cell_inds(self, x: int, y: int, z: int)->list[int]:
        """Returns the adjacent cells to the given grid point."""
        cells = []
        for dx in [-1, 0]:
            for dy in [-1, 0]:
                for dz in [-1, 0]:
                    if (
                        0 <= x + dx < self.num_cells[0]
                        and 0 <= y + dy < self.num_cells[1]
                        and 0 <= z + dz < self.num_cells[2]
                    ):
                        cells.append(self.encode_cell(x + dx, y + dy, z + dz))
        return cells



class SpatialMediumCreator:
    def __init__(self, box: td.Box, dl: float, wave_guide: td.Structure):
        self.box = box
        self.dl = dl
        self.wave_guide = wave_guide

    @classmethod
    def calc_cell_factor(cls, cell: CellBox, geometry: td.Geometry):
        q = queue.SimpleQueue()
        q.put(cell)
        filled_cells = []
        threshold = cell.size[0] / 2 ** ITERATIONS
        while not q.empty():
            tmp_cell = q.get()
            if not tmp_cell.intersects(geometry, strict_inequality=[True, True, True]):
                continue
            vertices = dict(zip("xyz", tmp_cell.vertices().T))
            if all(geometry.inside(**vertices)):
                filled_cells.append(tmp_cell)
            else:
                # split the tmp_cell if necessary
                if tmp_cell.size[0] > threshold:
                    for c in tmp_cell.split_evenly():
                        q.put(c)

        if not filled_cells:
            return 0

        v = cell.volume()
        assert v > 0  # noqa: S101

        return sum([c.volume() for c in filled_cells]) / v

    def create(self) -> td.CustomMedium:
        def linespace(min_val, max_val, dl):
            if min_val > max_val:
                raise ValueError("min_val must be less than or equal to max_val")
            num_points = int((max_val - min_val) / dl) + 1
            return np.linspace(min_val, max_val, num_points)

        def make_grid(lx, ly, lz, dl) -> CellGrid:
            x = linespace(-lx / 2, lx / 2, dl)
            y = linespace(-ly / 2, ly / 2, dl)
            z = linespace(-lz / 2, lz / 2, dl)
            boundaries = td.Coords(x=x, y=y, z=z)
            return CellGrid(boundaries=boundaries)

        def calc_cell_permittivity(cell: CellBox, wave_guide: td.Structure):
            geometry = wave_guide.geometry
            medium = wave_guide.medium
            eps = medium.eps_model(None)
            p = SpatialMediumCreator.calc_cell_factor(cell, geometry)
            return (1-p) * 1.0 + p * eps.real


        box = self.box
        wave_guide = self.wave_guide
        dl = self.dl

        bg_permittivity = wave_guide.background_permittivity or 1.
        cell_grid = make_grid(box.size[0], box.size[1], box.size[2], dl)
        cell_permittivity = np.full(cell_grid.n_cells(), bg_permittivity)
        total_cells = cell_grid.n_cells()
        print("total cells", total_cells)
        for idx in range(total_cells):
            x, y, z = cell_grid.decode_cell(idx)
            cell = cell_grid.cell_at(x, y, z)
            cell_permittivity[idx] = calc_cell_permittivity(
                cell, wave_guide) or bg_permittivity

        # initial grid points with cell permittivity
        boundaries = cell_grid.boundaries
        data = np.zeros((boundaries.x.size, boundaries.y.size, boundaries.z.size))
        for x in range(boundaries.x.size):
            for y in range(boundaries.y.size):
                for z in range(boundaries.z.size):
                    adj_cells = cell_grid.adjacent_cell_inds(x, y, z)
                    vals = [cell_permittivity[c] for c in adj_cells]
                    data[x, y, z] = np.average(vals)

        permittivity = SpatialDataArray(data, coords=boundaries.to_dict)
        return td.CustomMedium(permittivity=permittivity, conductivity=None)
