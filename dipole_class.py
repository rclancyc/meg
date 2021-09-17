import sys
sys.path.append('/Users/clancy/repos/meg')

import numpy as np
from numpy.linalg import norm, lstsq
from scipy import optimize
import yaml
from sensor_array_class import SensorArrayGenerator
from mesh_class import MeshGenerator

class DipoleLocalizer:
    def __init__(self, config_file, sensors=None, bias_field=None):
        """
        :param config_file: dict or str

        This class estimates the location of a dipole based on given data. It accepts either a configuration file path
        or a dictionary specifying the necessary settings. If an externally generated sensor array is provided, then
        the sensor location and sensor_directions must be specified. If the sensor direction is a single vector, all
        sensors with share that orientation
        """

        # this loads the yaml file (or dictionary), extracting settings and other parameters.
        if type(config_file) is str:
            configs = open(config_file)
            settings = yaml.load(configs, Loader=yaml.FullLoader)
        elif type(config_file) is dict:
            settings = config_file
        else:
            print('Setting file type not recognized, pass configuration file or dictionary with parameters')
            settings = None
        self.settings = settings
        self.model_type = self.settings['model']['model_type']
        self.sphere_radius = self.settings['model']['radius']  # in centimeters, appx avg of radius of human head

        # generate mesh from here
        mesh_obj = MeshGenerator(settings)
        self.mesh = mesh_obj.mesh

        # generate sensor array upon initialization (this object will track gradiometer sensors also)
        if sensors is None:
            self.sensor_obj = SensorArrayGenerator(settings)
        else:
            # NOTE THAT IF SENSORS AND BIAS FIELDS ARE INPUT INTO INITIALIZATION, THEY WILL OVER-RIDE ALL OTHER COMMANDS
            self.sensors = sensors
            bias_field_dir = bias_field / norm(bias_field)
            sensor_dir = np.tile(bias_field_dir, (self.sensors.shape[0], 1))

            self.sensor_directions = sensor_dir
            self.sensor_obj = type('obj', (object,), {'sensors': sensors, 'sensor_directions': sensor_dir})



        # only retain sensors/directions in primary (not secondary gradiometer) array
        # once we have more complicated gradiometry, we will need to improve the following logic,
        # that is, we can't just divide by 2 to find number of sensors, but it suffices for now.
        if 'num_sensors' not in settings['sensors'].keys():
            if settings['sensors']['gradiometers']:
                last_sens = int(self.sensor_obj.sensors.shape[0]/2)
            else:
                last_sens = self.sensor_obj.sensors.shape[0]
            self.settings['sensors']['num_sensors'] = last_sens
        else:
            last_sens = settings['sensors']['num_sensors']

        self.sensors = self.sensor_obj.sensors[0:last_sens,:]
        self.sensor_directions = self.sensor_obj.sensor_directions[0:last_sens,:]

        # class variables for dipole estimation
        self.crs_dipole_moment = None       # estimated dipole moment on mesh grid
        self.crs_dipole_location = None     # estimated dipole location mesh grid
        self.fne_dipole_moment = None       # estimated dipole moment after iterative method
        self.fne_dipole_location = None     # estimated dipole location after iterative method
        self.crs_res = None                 # residual of estimated dipole on mesh grid
        self.fne_res = None                 # residual of estimated dipole after iterative method
        self.res_on_mesh = None             # store residual of each best fit (p,q) on coarse mesh  USE THIS FOR OBJ
        self.location_scaling = None        # approximate norm of location
        self.moment_scaling = None          # approximate norm of moment

        # data that's provided to DipoleLocalizer class
        self.sensor_data = None                 # true magnetic field projected onto sensor axes plus noise
        self.sensor_data_norm = None            # store norm of data for appropriate scaling (might change someday)

        # self.lead_field_obj = None
        self.leads_for_mesh_calculated = False  # sets to true when lead fields are calculated...refreshes for new mesh
        self.lead_field_list = list()           # stores lead field matrices for all grid points.

        # initialize lead field object
        if settings['sensors']['gradiometers']:
            # THIS SECTION OF CODE IS A BIT OF A WORK AROUND. IF WE HAVE A GRADIOMETER SETUP, THEN THE MODEL TYPE WILL
            # CHANGE FROM WHAT IT WAS SET AS IN THE CONFIG FILE. NOT SURE OF THE BEST WAY TO PROCEED BUT THIS WILL
            # REQUIRE ATTENTION MOVING FORWARD AS GRADIOMETRY BECOMES MORE COMPLEX.
            if settings['sensors']['gradiometer_type'] == 'radial':
                # in the case of gradiometers, we need to use all sensors in the forward field model.
                self.model_type = 'sphere_with_gradiometers'
                self.lead_field_obj = LeadFieldGenerator(self.model_type,
                                                         self.sensor_obj.sensors, self.sensor_obj.sensor_directions)
        else:
            self.lead_field_obj = LeadFieldGenerator(self.model_type, self.sensors, self.sensor_directions)


    def huber_loss(self, dipole_guess):
        # scale dipole location down appropriately (otherwise step-size way too large
        c = 1.345
        loc_i = dipole_guess[0:3]/self.dip_loc_scale
        mom_i = dipole_guess[3:6]   # don't need to worry about this yet since the data will be scaled the same.
        y = self.sensor_data / self.sensor_data_norm
        est_field = self.get_magnetic_field(loc_i, mom_i)
        resid = est_field - y
        median_resid = np.median(np.abs(resid))
        tau = 1.5 * median_resid
        k = c*tau
        abs_resid = np.abs(resid)
        huber = np.sum(resid[abs_resid<=k]**2) + np.sum(2*k*abs_resid[abs_resid>k]-k**2)
        return huber


    ###################### NEW VERSION (old version in code graveyard)
    def scaled_residual(self, scaled_pq):
        """
        params scaled_pq: scaled estimated dipole location and moment stacked with close to unit norms
        this function calculates scaled_resid = (1/||est_B||)*||est_B - measured_B||
        """
        gamma = 1/self.sensor_data_norm         # this scales data in objective so its magnitude isn't too small
        alpha = self.location_scaling           # used to unscale location in residual for correct value
        beta = self.moment_scaling              # used to unscale moment in residual for correct value
        p_scaled = scaled_pq[:3]                # p_scaled should have norm close to 1 (not 1e-2 m)
        q_scaled = scaled_pq[3:]                # q_scaled should have norm close to 1 (not 1e-6 Am)

        y = self.sensor_data                                # raw sensor data
        L = self.calculate_lead_fields(alpha * p_scaled)    # p_original = alpha * p_scaled
        b = np.dot(L, beta*q_scaled)                        # q_original = beta * q_scaled
        scaled_resid = gamma * norm(y - b)                  # norm_y scales the residual
        return scaled_resid

    def improve_estimate(self):
        """
        This method scales the the step-sizes appropriately. In particular, it sets intial guesses for both p and q to
        have unit norms. The objectives return scaled residuals, not actual residuals (i.e. divided by the norm
        of sensor data for improved optimizer performance).

        If we fit a large erroneous q_crs for a distant p_crs, then this scaling could be an issue. For the moment, we
        leave as is and can address in the future for different use cases.
        """
        p_est = self.crs_dipole_location        # set local location estimate
        q_est = self.crs_dipole_moment          # set local moment estimate
        y = self.sensor_data                    # set local sensor data variable
        self.sensor_data_norm = norm(y)         # collect data norm so stopping criteria not hit too soon
        self.location_scaling = norm(p_est)     # set location scaling
        self.moment_scaling = norm(q_est)       # set moment scaling

        # following lines scale initialized data and pass scaled objective (unit vectors initially)
        pq_init = np.hstack((  p_est/self.location_scaling,   q_est/self.moment_scaling  ))
        better_est = optimize.minimize(self.scaled_residual, pq_init, method='BFGS', tol=1e-7, options={'eps': 1e-10})

        self.fne_dipole_location = better_est.x[0:3]*self.location_scaling  # unscale to get true location estimate
        self.fne_dipole_moment = better_est.x[3:6]*self.moment_scaling      # unscale to get true moment estimate
        L = self.calculate_lead_fields(self.fne_dipole_location)            # retrieve lead field for fine estimate
        b = np.dot(L, self.crs_dipole_moment)                               # calculate magentic field
        self.fne_res = norm(y - b)                                          # calculate true residual


    def estimate_dipole(self, refine=True):
        """
        This method will use the LEAD FIELDS and SENSOR DATA to find optimal dipole
        moments q_est for every possible dipole location.
        """
        min_res = np.infty
        p_best = None
        q_best = None
        if not self.leads_for_mesh_calculated:
            print("Calculating lead fields...this might take a little while")
            self.get_all_lead_fields()
        y = self.sensor_data
        self.res_on_mesh = np.zeros((len(self.mesh)))
        for i, p in enumerate(self.mesh):
            L = self.lead_field_list[i]
            q_est, _, _, _ = lstsq(L, y, rcond=None)
            res = norm((y - np.matmul(L, q_est)))
            self.res_on_mesh[i] = res
            if res < min_res:
                p_best = p
                q_best = q_est
                min_res = res
        self.crs_dipole_location = p_best
        self.crs_dipole_moment = q_best
        self.crs_res = min_res

        # unless otherwise specified, refine the estimate
        if refine:
            self.improve_estimate()


    def get_magnetic_field(self, p, q):
        """
        For a given dipole location and moment, calculate the magnetic field
        p: location of dipole
        q: dipole
        """
        L = self.calculate_lead_fields(p)
        B = np.dot(L, q)
        return B


    def get_all_lead_fields(self):
        lead_list = list()
        for p in self.mesh:                         # walk though all mesh points and calculate their lead fields
            l_p = self.calculate_lead_fields(p)
            lead_list.append(l_p)
        self.lead_field_list = lead_list
        self.leads_for_mesh_calculated = True


    def calculate_lead_fields(self, p):
        """
        This method is intended to use model type to extract appropriate lead field
        :param p: dipole location
        :return L: lead field from dipole located at p. Model to use passed from
        """
        L = self.lead_field_obj.get_lead_field(p)
        return L


    def load_sensor_data(self, sensor_data):
        self.sensor_data = sensor_data






















# LEAD FIELD GENERATOR CLASS ##########################################################################################
#######################################################################################################################
#######################################################################################################################
class LeadFieldGenerator:
    def __init__(self, model_type, sensors, sensor_directions):
        """
        :param model_type: specified in congif
        :param sensors: matrix with sensor locations
        :param sensor_directions: matrix with orientations of sensors.
        Change input to be model_type, sensor_locations, and sensor_directions

        This class is intended to serve as a centralized location for different lead field models. As geometry or models
        become more complex, this will allow for an easy to manage repository.

        :param dip_loc: Pass the entire DipoleLocalizer object for which we want lead fields
        :return: void
        """
        # extract essential variables from dipole localizer object to use in following methods.
        self.model_type = model_type
        self.sensors = sensors
        self.sensor_directions = sensor_directions

    def get_lead_field(self, p):
        """
        This method uses model type as specified at initialization to calculate appropriate lead field.
        :param p: dipole location
        :return: lead field for given model generated by dipole located at p
        """
        model = self.model_type
        sensor_locs = self.sensors
        sensor_dirs = self.sensor_directions
        if model == 'conducting_sphere':
            L = self.conducting_sphere_lead_fields(p, sensor_locs, sensor_dirs)
        elif model == 'free_space_magnetic':
            L = self.free_space_magnetic_dipole_lead_fields(p, sensor_locs, sensor_dirs)
        elif model == 'free_space_current':
            L = self.free_space_current_dipole_lead_fields(p, sensor_locs, sensor_dirs)
        elif model == 'sphere_with_gradiometers':
            L = self.radial_gradio_conducting_sphere_lead_fields(p, sensor_locs, sensor_dirs)
        ### JUST ADDED
        elif model == 'radial_gradio_conducting_sphere_lead_fields':
            L = self.radial_gradio_conducting_sphere_lead_fields(p, sensor_locs, sensor_dirs)
        ### END JUST ADDED
        return L

    def get_full_lead_field(self, p):
        """
        This method uses model type as specified at initialization to calculate appropriate FULL lead fields. FULL lead
        fields is the magnetic field in all three coordinate axes.
        :param p: dipole location
        :return: lead field for given model generated by dipole located at p. Must reshape when passed back to caller
        """
        model = self.model_type
        sensor_locs = self.sensors
        n_sensors = len(sensor_locs)
        full_locs = np.zeros((3*n_sensors, 3))
        full_dirs = np.zeros((3 * n_sensors, 3))

        for i, sens in enumerate(sensor_locs):
            full_locs[(3*i):(3*(i+1)), :] = np.tile(sens, (3,1))
            full_dirs[(3*i):(3*(i+1)), :] = np.eye(3)

        if model == 'conducting_sphere':
            L = self.conducting_sphere_lead_fields(p, full_locs, full_dirs)
        elif model == 'free_space_magnetic':
            L = self.free_space_magnetic_dipole_lead_fields(p, full_locs, full_dirs)
        elif model == 'free_space_current':
            L = self.free_space_current_dipole_lead_fields(p, full_locs, full_dirs)
        elif model == 'radial_gradio_conducting_sphere_lead_fields':
            L = self.radial_gradio_conducting_sphere_lead_fields(p, full_locs, full_dirs)
        else:
            print('Unrecognized model type, please specify "free_space" or "conducting_sphere" in configuration file')
            L = None
        return L

    @staticmethod
    def free_space_magnetic_dipole_lead_fields(p, sensor_locations, sensor_directions):
        """
        :param p: (3,) numpy array
        :param sensor_locations: (n_sensors, 3) numpy array
        :param sensor_directions: (n_sensors, 3) numpy array

        This function calculates the lead field matrix at all sensors. No dependence on dipole strength or orientation
        NOTE: This is the linear operator that acts on a MAGNETIC dipole, not a current dipole as the others
        """
        c = 1e-7
        n_sensors = len(sensor_locations)
        r = sensor_locations
        D = r - p

        i = 0
        L = np.zeros((n_sensors, 3))
        for d, s in zip(D, sensor_directions):
            L[i, :] = c *( (np.dot(s, d) * d.T)/norm(d)**5 -  s.T/norm(d)**5)
            i += 1
        return L


    @staticmethod
    def free_space_current_dipole_lead_fields(p, sensor_locations, sensor_directions):
        """
        :param p: (3,) numpy array
        :param sensor_locations: (n_sensors, 3) numpy array
        :param sensor_directions: (n_sensors, 3) numpy array

        This function calculates the lead field matrix at all sensors. No dependence on dipole strength or orientation
        """
        c = 1e-7
        n_sensors = len(sensor_locations)
        r = sensor_locations
        d = p - r
        d_norm = norm(d, axis=1)

        L = np.zeros((n_sensors, 3))
        for i in range(n_sensors):
            C_d = np.array( [[0,        -d[i,2],    d[i,1]  ],
                             [d[i,2],   0,          -d[i,0] ],
                             [-d[i,1],  d[i,0],     0       ]])
            K = (c/d_norm[i]**3)*C_d

            if d_norm[i] == 0:
                i = i
                print('A mesh point is located on a sensor, change to avoid division by zero')

            # construct lead field matrix one sensor at a time
            L[i, 0:3] = np.dot(sensor_directions[i, :], K)
        return L


    @staticmethod
    def conducting_sphere_lead_fields(p, sensor_locations, sensor_directions):
        """
        params p: dipole location
        This function calculates the lead field matrix at all sensors due to a dipole at point p.
        No dependence on dipole strength or orientation
        """
        c = 1e-7
        n_sensors = sensor_locations.shape[0]
        r = sensor_locations
        d = r - p
        p_mat = p * np.ones((len(r), 3))
        r_norm = norm(r, axis=1)
        d_norm = norm(d, axis=1)
        d_dot_r = np.sum(np.multiply(d, r), axis=1)
        C_p = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        F = d_norm * (r_norm * d_norm + r_norm ** 2 - np.matmul(r, p))
        grad_F = ((d_norm ** 2 / r_norm + d_dot_r / d_norm + 2 * d_norm + 2 * r_norm) * r.T
                  - (d_norm + 2 * r_norm + d_dot_r / d_norm) * p_mat.T)
        grad_F = grad_F.T
        L = np.zeros((n_sensors, 3))
        for i in range(n_sensors):
            sensor_dir = sensor_directions[i, :]
            K = c * np.matmul((np.outer(grad_F[i], r[i]) - F[i] * np.identity(3)) / F[i] ** 2, C_p)
            L[i, 0:3] = np.matmul(sensor_dir, K)

        return L

    @staticmethod
    def radial_gradio_conducting_sphere_lead_fields(p, sensor_locations, sensor_directions):
        """
        params p: dipole location
        This function calculates the lead field matrix at all sensors due to a dipole at point p.
        No dependence on dipole strength or orientation
        """
        c = 1e-7
        n_sensors = sensor_locations.shape[0]
        r = sensor_locations
        d = r - p
        p_mat = p * np.ones((len(r), 3))
        r_norm = norm(r, axis=1)
        d_norm = norm(d, axis=1)
        d_dot_r = np.sum(np.multiply(d, r), axis=1)
        C_p = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
        F = d_norm * (r_norm * d_norm + r_norm ** 2 - np.matmul(r, p))
        grad_F = ((d_norm ** 2 / r_norm + d_dot_r / d_norm + 2 * d_norm + 2 * r_norm) * r.T
                  - (d_norm + 2 * r_norm + d_dot_r / d_norm) * p_mat.T)
        grad_F = grad_F.T
        L = np.zeros((n_sensors, 3))
        for i in range(n_sensors):
            sensor_dir = sensor_directions[i, :]
            K = c * np.matmul((np.outer(grad_F[i], r[i]) - F[i] * np.identity(3)) / F[i] ** 2, C_p)
            L[i, 0:3] = np.matmul(sensor_dir, K)

        n_channels = int(n_sensors/2)
        L = L[:n_channels,:] - L[n_channels:,:]

        return L
