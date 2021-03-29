import numpy as np
from deepspace.config.config import config, logger
from dxray.astra.projection import project


class projections:
    def __init__(self, mesh, max_projections=3, resolution=(512, 512)) -> None:
        self.mesh = mesh
        self.shape = (max_projections) + resolution
        self.projections = np.zeros(self.shape)
        self.max_projections = max_projections
        self.resolution = resolution
        self.current_projection = 0

    def set_mesh(self, mesh):
        self.mesh = mesh

    def reset(self) -> None:
        self.projections = np.zeros(self.shape)

    def project(self, mesh, angle):
        this_projection = project(mesh, angles=angle, heights=None, settings=config.environment)
        self.projections[self.current_projection] = this_projection
        self.current_projection += 1
