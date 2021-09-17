import numpy as np
import pandas as pd
from numpy.linalg import norm
from helper_functions import get_cartesian_coords, get_spherical_coords



# SENSOR ARRAY GENERATOR CLASS ########################################################################################
#######################################################################################################################
#######################################################################################################################
class SensorArrayGenerator:
    def __init__(self, settings):
        self.sensors = None
        self.sensor_directions = None
        self.settings = settings
        self.sphere_radius = settings['model']['radius']

        array_type = settings['sensors']['array_type']
        if 'density' not in settings['sensors']:
            settings['sensors']['density'] = False

        if settings['sensors']['density']:
            self.create_density_array()
        else:
            if array_type == 'fibonacci':
                self.create_fibonacci_array()
            elif array_type == 'external':
                self.create_from_file_array()
            elif array_type == 'visualization':
                self.create_visualization_array()
            elif array_type == 'uniform':
                self.create_uniform_array()
            elif array_type == 'spherical':
                self.create_spherical_array()
            elif array_type == 'sensor_patch':
                self.create_patch_array()
            elif array_type == 'phantom':
                self.phantom_sensors()
            elif array_type == 'tunneling_wall':
                self.create_tunneling_wall_array()

        if settings['sensors']['gradiometers']:
            self.add_gradiometer_sensors()

    def create_from_file_array(self):
        s = self.settings                           # load settings
        bias_field = s['sensors']['bias_field']
        array_file = s['sensors']['array_file']     # extract sensor file
        raw_array = pd.read_csv(array_file)          # get raw sensor data
        use_sensors = s['sensors']['use_sensors']   # determine which sensors to use
        if use_sensors == 'all':
            sensor_array = raw_array
        else:
            idx = raw_array['sensor_number'].isin(use_sensors)
            sensor_array = raw_array[idx]

        sensors = np.asarray(sensor_array[['x', 'y', 'z']])

        if 'x_orientation' in raw_array.columns:
            sensor_dir = np.asarray(sensor_array[['x_orientation', 'y_orientation', 'z_orientation']])
        elif bias_field == 'radial':
            sensor_dir = sensors
        else:
            sensor_dir = np.tile(bias_field, (sensors.shape[0],1))

        sensor_dir = (sensor_dir.T / norm(sensor_dir, axis=1)).T
        self.sensors = sensors
        self.sensor_directions = sensor_dir


    def create_visualization_array(self):
        # generate generic mesh grid with x, y, and z values
        s = self.settings
        model = s['model']['model_type']
        mesh_center = s['model']['center']
        lvl = s['mesh']['level']
        radius = self.sphere_radius
        bias_field = s['sensors']['bias_field']
        if bias_field is not None:
            bias_field = bias_field/norm(bias_field)

        ext = (radius if model == 'conducting_sphere' else 0.1*radius)
        pts = np.linspace(-radius*(1 + ext), radius*(1 + ext), 2**lvl)
        xs = mesh_center[0] + pts
        ys = mesh_center[1] + pts
        zs = mesh_center[2] + pts
        if model == 'conducting_sphere':
            zs = zs - min(zs)  # shift up to grid is centered on top of sphere.

        mesh = list()
        for x in xs:
            for y in ys:
                for z in zs:
                    mesh_pt = np.array([x,y,z])
                    if model == 'conducting_sphere':
                        if norm(mesh_pt) > radius and z >= 0:
                            mesh.append(mesh_pt)
                    elif model == 'free_space':
                        mesh.append(mesh_pt)

        self.sensors = np.asarray(mesh)
        self.sensor_directions = np.tile(bias_field, (len(mesh), 1))


    def create_uniform_array(self):
        s = self.settings
        n_sensors = s['sensors']['num_sensors']
        bias_field = s['sensors']['bias_field']
        radius = self.sphere_radius
        thetas_phis = np.zeros((n_sensors,2))

        # randomly draw theta from unif(0,2pi) and phi using inverse transfom sampling to push values away from top
        thetas_phis[:,0] = np.random.uniform(0, 2 * np.pi, (n_sensors,))
        temp = np.random.uniform(0, 1, (n_sensors,))
        thetas_phis[:, 1] = np.pi / 2 * temp**(1/2)
        sensor_locs = np.zeros((n_sensors,3))
        for i, tp in enumerate(thetas_phis):
            sensor_locs[i,:] = get_cartesian_coords(radius, tp[0], tp[1])

        if bias_field is None:
            sensor_dir = (sensor_locs.T / norm(sensor_locs, axis=1)).T
        else:
            bias_field = bias_field/norm(bias_field)
            sensor_dir = np.tile(bias_field, (n_sensors, 1))

        self.sensors = sensor_locs
        self.sensor_directions = sensor_dir


    def create_fibonacci_array(self):
        """
        This is based on the fibonacci sphere and code is pulled almost directly from
        Fnord's post at:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        """
        s = self.settings
        n_sensors = s['sensors']['num_sensors']
        bias_field = s['sensors']['bias_field']
        radius = self.sphere_radius

        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(2*n_sensors):
            y = 1 - (i / float(2*n_sensors - 1)) * 2  # y goes from 1 to -1
            r = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            if z >= 0 and len(points) < n_sensors:
                points.append((x, y, z))

        sensor_locs = np.asarray(points) * radius

        if isinstance(bias_field, str) or bias_field is None:
            # CONDITIONAL MIGHT HAVE UNEXPECTED BEHAVIORx
            sensor_dir = (sensor_locs.T / norm(sensor_locs, axis=1)).T
        else:
            bias_field = bias_field / norm(bias_field)
            sensor_dir = np.tile(bias_field, (n_sensors, 1))

        self.sensors = sensor_locs
        self.sensor_directions = sensor_dir


    def create_spherical_array(self):
        """
        This method constructs a spherical sensor array. We might want to add
        an offset (i.e. sensors should be 5mm above surface of scalp but we can
        worry about that a little later).
        """
        s = self.settings['sensors']
        n_thetas = s['num_thetas']
        n_phis = s['num_phis']
        bias_field = s['bias_field']
        radius = self.sphere_radius
        thetas = np.linspace(0, (n_thetas - 1) * 2 * np.pi / n_thetas, n_thetas)
        phis = np.linspace(.5 * np.pi / n_phis, .5 * np.pi, n_phis)
        sensor_array = np.zeros((len(thetas) * len(phis), 3))
        sensor_dir = np.zeros((len(thetas) * len(phis), 3))
        counter = 0

        for phi in phis:
            for theta in thetas:
                temp = get_cartesian_coords(radius, theta, phi)
                sensor_array[counter, :] = temp
                if bias_field is None:
                    sensor_dir[counter, :] = temp / norm(temp)
                else:
                    sensor_dir[counter, :] = bias_field
                counter += 1

        self.sensors = sensor_array
        self.sensor_directions = sensor_dir


    def create_patch_array(self):
        s = self.settings['sensors']
        phi_center = s['phi_center']
        phi_range = s['phi_range']
        if phi_range < 5:
            print('Inputs should be in degrees, please confirm appropriate phi range')
        n_phis = s['num_phis']
        n_thetas = s['num_thetas']
        bias_field = s['bias_field']
        radius = self.sphere_radius

        # convert angles to radians for calculations.
        phi_center = np.pi*phi_center/180
        phi_range = np.pi*phi_range/180

        # generate theta space, remove 2*pi entry
        thetas = np.linspace(0, 2*np.pi, n_thetas+1)
        thetas = thetas[0:-1]

        # generate phi space, prevent from placing entry on top of the spheere
        phis = np.linspace(0,  phi_range, n_phis+1)
        phis = phis[1::]

        # initialize sensor arrays
        sensor_array = np.zeros((len(thetas) * len(phis), 3))
        sensor_dir = np.zeros((len(thetas) * len(phis), 3))

        # create rotation matrix
        w = phi_center
        R = np.array([[ np.cos(w), 0, np.sin(w)],
                      [         0, 1, 0        ],
                      [-np.sin(w), 0, np.cos(w)]])
        counter = 0

        # generate sensor locations based on grid then rotate into appropriate place
        for phi in phis:
            for theta in thetas:
                sensor_array[counter, :] = np.dot(R, get_cartesian_coords(radius, theta, phi)) # rotate coord by phi_center angle
                counter += 1

        # generate sensor orientations
        if bias_field is (None or 'radial'):
            for i, sens in enumerate(sensor_array):
                sensor_dir[i, :] = sens / norm(sens)
        else:
            bias_field_dir = bias_field / norm(bias_field)
            sensor_dir = np.tile(bias_field_dir, (sensor_array.shape[0], 1))

        self.sensors = sensor_array
        self.sensor_directions = sensor_dir


    def create_density_array(self):
        ### NEXT TIME WE USE THIS METHOD, NEED TO REWORK IT.
        # determine type of sensor array then create it.
        s = self.settings['sensors']
        n_sensors_included = s['num_sensors_included']
        total_n_sensors = s['total_num_sensors']
        phi_center = s['phi_center']
        bias_field = s['bias_field']
        array_type = s['array_type']

        if array_type == 'fibonacci':
            self.create_fibonacci_sensor_array(total_n_sensors)
        elif array_type == 'uniform':
            self.create_uniform_sensor_array(total_n_sensors, bias_field)
        elif array_type == 'spherical':
            temp = int(np.ceil(np.sqrt(total_n_sensors)))
            self.create_spherical_sensor_array(temp, temp)
        else:
            print('Sensor array type not recognized, choose from spiral, uniform, or spherical.')

        # find distance from patch center to all sensors then find sorted index
        r0 = get_cartesian_coords(self.sphere_radius, 0, phi_center)
        dist = np.zeros(self.sensors.shape[0])
        for i, s in enumerate(self.sensors):
             dist[i] = norm(s - r0)
        idx = dist.argsort()

        # sort sensors with nearest at top
        self.sensors = self.sensors[idx]
        self.sensor_directions = self.sensor_directions[idx]

        self.sensors = self.sensors[0:n_sensors_included,:]
        self.sensor_directions = self.sensor_directions[0:n_sensors_included,:]


    def load_phantom_sensors(self, use_sensors = [103, 109, 117, 201, 209, 217, 301, 307, 317]):
        if use_sensors == 'all':
            use_sensors = [101, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117,
                           201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                           301, 302, 303, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316, 317]

        bias_field = self.settings['sensors']['bias_field']
        raw_sensors = pd.read_csv('/meshes/phantom_sensors.csv')
        idx = raw_sensors['sensor_number'].isin(use_sensors)
        raw_sensors = raw_sensors[idx]

        self.sensors = raw_sensors
        if bias_field == 'radial':
            self.sensor_directions =  (raw_sensors.T/norm(raw_sensors, axis=1)).T  # all sensor dirs are radial for phantom
        else:
            bias_field_dir = bias_field / norm(bias_field)
            self.sensor_directions = np.tile(bias_field_dir, (self.sensors.shape[0], 1))

        self.sensor_indices = idx


    def create_tunneling_wall_array(self):
        # hard code sensors, this will need to change in the future.
        bias_field = self.settings['sensors']['bias_field']
        self.sensors = np.array([[-10,0,0], [0,0,0], [10,0,0], [-10,0,2], [0,0,2], [10,0,2]])
        bias_field_dir = bias_field / norm(bias_field)
        sensor_dir = np.tile(bias_field_dir, (self.sensors.shape[0], 1))
        self.sensor_directions = sensor_dir


    def add_gradiometer_sensors(self):
        """
        Method will add new sensors in radial direction with sensor orientation matching corresponding sensor.
        The number of sensors will double when this method is called.
        :return: void
        """
        gradiometer_offset = self.settings['sensors']['gradiometer_offset']
        s = self.sensors
        rad_dir = (s.T/norm(s.T, axis=0)).T
        grad_sensors = s + gradiometer_offset * rad_dir
        self.sensors = np.vstack((self.sensors, grad_sensors)) # stack old sensors and new sensor locations for gradiom
        self.sensor_directions = np.tile(self.sensor_directions, (2,1)) # field should be in same direction I imagine
        print('There are now a total of', self.sensors.shape[0], 'sensors in the array due to gradiometry.')
