"""
X-Ray evironment for deep reinforcemnet
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from deepspace.config.config import config, logger

from dxray.astra.projection import project
from dxray.astra.utils import fix_mesh
from dxray.astra.modify import random_break


class Astra:
    """Astra simulation for mesh projection
    """

    def __init__(self, max_projections=3, resolution=(512, 512)) -> None:
        # self._angle = self.init_angle()
        # self._projections = np.zeros(self.shape)
        # self._current_projection = 0
        # self.broken_mesh = None

        self.shape = (max_projections, ) + resolution
        self.max_projections = max_projections
        self.resolution = resolution
        self._mesh = self.load_mesh()

        self.init_break()
        self.reset()

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh):
        self._mesh = new_mesh

    @property
    def projections(self):
        return self._projections

    @property
    def current_projection(self):
        return self._current_projection

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, new_angle):
        self._angle = new_angle

    @property
    def quaternion(self):
        # transfer the angles into quaterions
        return R.from_euler('yxz', self.angle, degrees=True).as_quat()

    def reset(self) -> None:
        self.angle = self.init_angle()
        self._current_projection = 0
        self._projections = np.zeros(self.shape)
        self.broken_mesh = self.mesh
        # project once
        self.break_mesh()
        self.project()

    def project(self):
        this_projection = project(self.broken_mesh, angles=[self.quaternion], heights=None, settings=config.environment)
        self._projections[self._current_projection] = this_projection
        self._current_projection += 1

    def rotate(self, angle):
        self._angle += angle
        self._angle[2] = (self._angle[2] + angle) % config.environment.angle_range[2][1]

    def init_break(self):
        settings = config.environment
        # calculate for the position and radius of the defect
        mesh_min = self._mesh.vertices.min(axis=0)
        mesh_max = self._mesh.vertices.max(axis=0)
        config.swap = {}
        if settings.unit == 'scale':
            config.swap.center_range = [[mesh_min[i], mesh_max[i]] for i in range(3)]
            if 'radius_range' in settings:
                config.swap.radius_range = (self._mesh.scale * settings.radius_range[0], self._mesh.scale * settings.radius_range[1])
            else:
                config.swap.radius_range = (self._mesh.scale * 0.01, self._mesh.scale * 0.1)
            if 'remove_range' in settings:
                config.swap.remove_range = (self._mesh.vertices.shape[0] * settings.remove_range[0], self._mesh.vertices.shape[0] * settings.remove_range[1])
            else:
                config.swap.remove_range = (self._mesh.vertices.shape[0] * 0.01, self._mesh.vertices.shape[0] * 0.1)
        elif settings.unit == 'voxel':
            config.swap.center_range = [[mesh_min[i], mesh_max[i]] for i in range(3)]
            if 'radius_range' in settings:
                config.swap.radius_range = (settings.radius_range[0], settings.radius_range[1])
            else:
                config.swap.radius_range = (self._mesh.scale * 0.01, self._mesh.scale * 0.1)
            if 'remove_range' in settings:
                config.swap.remove_range = (settings.remove_range[0], settings.remove_range[1])
            else:
                config.swap.remove_range = (self._mesh.vertices.shape[0] * 0.01, self._mesh.vertices.shape[0] * 0.1)
        config.swap.get_removed_part = True

    def break_mesh(self):
        config.environment.swap = config.swap
        defect_mesh, vertices_mask, face_mask, removed_mesh = random_break(self._mesh, config.environment)
        self.broken_mesh = defect_mesh

    @staticmethod
    def load_mesh():
        # load mesh, fix watertight if necessery
        mesh = trimesh.load(config.settings.mesh_file_path)
        if not config.environment.watertight_cad:
            # fix mesh to water tight
            mesh = fix_mesh(mesh)
        return mesh

    @staticmethod
    def init_angle():
        angles = np.random.rand(3)
        # scale the random number into ranges on each direction
        angles = [angles[index] * (config.environment.angle_range[index][1] - config.environment.angle_range[index][0]) + config.environment.angle_range[index][0] for index in range(3)]
        return angles
