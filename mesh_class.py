import numpy as np
from numpy.linalg import norm
from helper_functions import get_cartesian_coords, get_spherical_coords

# MESH GENERATOR CLASS ################################################################################################
#######################################################################################################################
#######################################################################################################################
class MeshGenerator:
    def __init__(self, settings):
        """
        This class generates meshes. the argument
        """
        self.mesh = None
        self.settings = settings
        model_type = settings['model']['model_type']
        mesh_type = settings['mesh']['mesh_type']

        if model_type == 'conducting_sphere':
            self.sphere_radius = settings['model']['radius']
            if mesh_type == 'cartesian_inner_sphere':
                self.generate_cartesian_mesh_for_conducting_sphere()
            elif mesh_type == 'spherical_inner_sphere':
                self.generate_spherical_mesh()

        if mesh_type == 'box_ROI' or mesh_type == 'ROI':
            self.generate_cartesian_mesh_for_box()

        if mesh_type == 'sphere_ROI':
            self.generate_cartesian_mesh_for_sphere()

        if mesh_type == 'cylinder_temp':
            self.generate_cartesian_mesh_for_cylinder_temp()

        if mesh_type == 'cylinder_temp2':
            self.generate_cartesian_mesh_for_cylinder_temp2()


    def generate_cartesian_mesh_for_sphere(self):
        """
        Generates a cartesian mesh on the upper half the interior of the spheres
        """
        s = self.settings['mesh']
        mesh_center = s['mesh_center']
        spacing = s['spacing']
        outer_radius = s['outer_radius']
        xs = np.arange(-outer_radius + mesh_center[0], outer_radius + mesh_center[0] + 1e-12, spacing)
        ys = np.arange(-outer_radius + mesh_center[1], outer_radius + mesh_center[1] + 1e-12, spacing)
        zs = np.arange(-outer_radius + mesh_center[2], outer_radius + mesh_center[2] + 1e-12, spacing)
        if 'inner_radius' in s.keys():
            inner_radius = s['inner_radius']
        else:
            inner_radius = 0

        mesh = []
        for x in xs:
            for y in ys:
                for z in zs:
                    mesh_point = np.array([x, y, z])
                    v = norm(mesh_point - mesh_center)
                    if inner_radius <= v < outer_radius:
                        mesh.append(mesh_point)
        self.mesh = np.asarray(mesh)


    def generate_spherical_mesh(self):
        """
        This method will generate a spherical mesh used to perform a grid search
        """
        m = self.settings['mesh']
        n_rad = m['num_radial']
        n_pol = m['num_polar']
        n_azi = m['num_azimuthal']
        min_radius_factor = m['min_radius_ratio']

        mesh = np.zeros(((n_rad * n_pol * n_azi), 3))  # initialize mesh
        pol_end = 2 * np.pi - 2 * np.pi / n_pol  # prevent reusing 2pi and pi for pol and azi resp.
        azi_end = np.pi - np.pi / n_azi

        # generate linear grids along each dimentions
        rads = np.linspace(min_radius_factor * self.sphere_radius, ((n_rad - 1) / n_rad) * self.sphere_radius, n_rad)
        pols = np.linspace(0, pol_end, n_pol)
        azis = np.linspace(0, azi_end, n_azi)

        # loop through all entries and transform to cartesian coordinates.
        counter = -1
        for r in rads:
            for p in pols:
                for a in azis:
                    counter += 1
                    # Not sure if the following function call will work
                    mesh[counter, :] = get_cartesian_coords(r, p, a)
        self.mesh = mesh

    def generate_cartesian_mesh_for_conducting_sphere(self):
        """
        Generates a cartesian mesh on the upper half the interior of the spheres
        """
        lvl = self.settings['mesh']['level']
        n_points = 2 ** lvl + 1
        inner_radius = self.sphere_radius * 0.98
        line_points = np.linspace(-inner_radius, inner_radius, n_points)

        mesh = []
        for x in line_points:
            for y in line_points:
                for z in line_points:
                    mesh_point = np.array([x, y, z])
                    if norm(mesh_point) <= self.sphere_radius and z >= -1.0e-8:
                        mesh.append(mesh_point)
        self.mesh = np.asarray(mesh)

    def generate_cartesian_mesh_for_box(self):
        """
        Generate cartesian mesh over specified region of interest.
        """
        m    = self.settings['mesh']
        if 'level' in m.keys():
            lvl  = m['level']
            n_points = 2 ** lvl + 1
            n_x = n_points
            n_y = n_points
            n_z = n_points
        else:
            n_x = m['num_x']
            n_y = m['num_y']
            n_z = m['num_z']
        xmin = m['xroi'][0]
        xmax = m['xroi'][1]
        ymin = m['yroi'][0]
        ymax = m['yroi'][1]
        zmin = m['zroi'][0]
        zmax = m['zroi'][1]

        mesh = []
        for x in np.linspace(xmin, xmax, n_x):
            for y in np.linspace(ymin, ymax, n_y):
                for z in np.linspace(zmin, zmax, n_z):
                    mesh_point = np.array([x, y, z])
                    mesh.append(mesh_point)
        self.mesh = np.asarray(mesh)

    def generate_cartesian_mesh_for_cylinder_temp(self):
        """
        THIS IS NOT MEANT TO BE A PERMANENT MESH GENERATE. NEED TO GENERALIZE IF THAT'S WHAT WE WANT FOR FLEXIBILITY
        """
        s = self.settings['mesh']
        ep1 = s['endpoint1']
        ep2 = s['endpoint2']
        spacing = s['spacing']
        outer_radius = s['outer_radius']
        xs = np.arange(ep1[0]-outer_radius, ep2[0]+outer_radius + 1e-12, spacing)
        ys = np.arange(-outer_radius, outer_radius + 1e-12, spacing)
        zs = np.arange(-outer_radius, outer_radius + 1e-12, spacing)
        if 'inner_radius' in s.keys():
            inner_radius = s['inner_radius']
        else:
            inner_radius = 0

        mesh = []
        for x in xs:
            for y in ys:
                for z in zs:
                    mesh_point = np.array([x, y, z])
                    if ep1[0] <= x <= ep2[0]:
                        if inner_radius <= np.sqrt(y**2+z**2) < outer_radius:
                            mesh.append(mesh_point)
                    else:
                        v1 = norm(mesh_point-ep1)
                        v2 = norm(mesh_point-ep2)
                        if inner_radius <= v1 < outer_radius or inner_radius <= v2 < outer_radius:
                            mesh.append(mesh_point)

        self.mesh = np.asarray(mesh)



    def generate_cartesian_mesh_for_cylinder_temp2(self):
        """
        THIS IS NOT MEANT TO BE A PERMANENT MESH GENERATE. NEED TO GENERALIZE IF THAT'S WHAT WE WANT FOR FLEXIBILITY
        """
        s = self.settings['mesh']
        ep1 = s['endpoint1']
        ep2 = s['endpoint2']
        spacing = s['spacing']
        outer_radius = s['outer_radius']
        xs = np.arange(ep1[0]-outer_radius, ep2[0]+outer_radius + 1e-12, spacing)
        ys = np.arange(-outer_radius, outer_radius + 1e-12, spacing)
        zs = np.arange(-outer_radius, outer_radius + 1e-12, spacing)
        if 'inner_radius' in s.keys():
            inner_radius = s['inner_radius']
        else:
            inner_radius = 0

        mesh = []
        for x in xs:
            for y in ys:
                for z in zs:
                    mesh_point = np.array([x, y, z])
                    if ep1[0] <= x <= ep2[0]:
                        if inner_radius <= np.sqrt(y**2+z**2) < outer_radius:
                            mesh.append(mesh_point)
        self.mesh = np.asarray(mesh)