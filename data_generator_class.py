import numpy as np
from numpy.linalg import norm
import copy
from helper_functions import get_cartesian_coords, get_spherical_coords
from dipole_class import LeadFieldGenerator


class DataGenerator:
    """
    This class is meant to generate data for the DipoleLocalizer class. Since data generated can be complex to
    manufacture, we store all the necessary code here. To run properly, a properly confired DipoleLocalizer instance
    must be provided as an argument. In particular sensor locations, sensor directions, and the model type will all be
    loaded up front to provide necessary information for data generation

    I envision the following structure:
        1.  Initialize with 'DipoleLocalizer' object including sensor_locs, sensors_dirs, and model
        2.  The DataGenerator class will store the information and use it to synthesize data
        3.  Either load external dipole OR call method to randomly generate it. This can be based on ROI and other input
        4.  With a dipole in hand, we can now generate data. Use must specify whether they want data generated using a
            scalar or vector magnetometers. Ideally, we will generate true sensor data, noisy sensor data, and full
            field data at this point. These values might depend on perturbations which must be given by the user. It's
            also reasonable that the user could specify amount of noise to add.
        5.  Provide user with methods to change sensor data without drawing a new dipole such as loading noise profile
            from elsewhere.
    """
    def __init__(self, dip_loc):
        self.settings = dip_loc.settings
        self.dip_loc = dip_loc
        self.sensor_data = None                         # sensor readings (generated internally)
        self.clean_sensor_data = None                   # generate field from a particular dipole
        self.dipole_location = None                     # location of dipole generator within class or passed in
        self.dipole_moment = None                       # moment of dipole generator within class or passed in
        self.full_bfield = None                         # generate a full magnetic for sensor locations
        self.sensor_dirs = dip_loc.sensor_directions    # retrieve sensor orientations from dipole_localizer
        self.sensor_locs = dip_loc.sensors              # retrieve sensor locations from dipole_localizer
        self.sensor_loc_perts = None                    # perturbations in sensor locations (if desired).
        self.sensor_dir_perts = None                    # perturbations in sensor directions (if desired)
        self.lead_field_obj = dip_loc.lead_field_obj    # this talks to lead field object instantiated in dip_loc
        self.rms_noise = dip_loc.settings['data']['rms_noise']
        self.moment_magnitude = dip_loc.settings['data']['moment_magnitude']
        self.bias_field = dip_loc.settings['sensors']['bias_field']
        self.model_type = dip_loc.settings['model']['model_type']
        if self.model_type == 'conducting_sphere':
            self.sphere_radius = dip_loc.sphere_radius

        # Note that the full sensor array is stored in the SensorArrayGenerator Object within the
        # DipoleLocalizer object. We are extracting it here
        self.full_sensors = self.dip_loc.sensor_obj.sensors
        self.full_dirs = self.dip_loc.sensor_obj.sensor_directions
        self.gradiometry_lead_field_obj = None          # we will generate this object when grad method called.
        self.pre_clean = None                           # true meas. at ALL sensors before noise orsubtracting gradiom.


    # loading data in for data
    def load_external_dipole(self, p, q):
        # should this just be passed in from elsewhere?
        self.dipole_location = p
        self.dipole_moment = q


    def load_sensor_location_perturbations(self, sensor_loc_perts):
        self.sensor_loc_perts = sensor_loc_perts


    def load_sensor_direction_perturbations(self, sensor_dir_perts):
        self.sensor_dir_perts = sensor_dir_perts


    # create dipoles
    def create_dipole_for_sphere(self):
        """
        Randomly draws dipole location and moment
        """
        depth_range = self.settings['data']['depth_range']
        inner_radius = depth_range[0]
        outer_radius = depth_range[1]
        dipole_scale = self.moment_magnitude
        theta = np.random.uniform(0, 2 * np.pi)  # any polar angle
        unif = np.random.uniform(0,1)
        phi = np.pi/2 * np.sqrt(unif) # use inverse CDF sampling here.
        r = np.random.uniform(inner_radius * self.sphere_radius, outer_radius * self.sphere_radius)  # not too deep in brain
        self.dipole_location = get_cartesian_coords(r, theta, phi)
        temp = np.random.normal(0, 2, 3)
        if self.settings['data']['moment_orientation'] == 'tangential':
            phat = self.dipole_location / norm(self.dipole_location)
            temp = temp - np.dot(temp, phat)*phat
        self.dipole_moment = dipole_scale*(temp / norm(temp))  # scale dipole


    # create dipoles
    def create_dipole_cartesian_gaussian(self, mu, sigma):
        """
        :param mu: ndarry (3 component mean in cartesian (x,y,z) )
        :param sigma: ndarray (3 component standard deviation )
        generates a random dipole from a standard normal in the region specified by
        """
        self.dipole_location = np.random.normal(mu, sigma)
        temp = np.random.normal(0,1,(3,))
        if self.settings['data']['moment_orientation'] == 'tangential':
            phat = self.dipole_location / norm(self.dipole_location)
            temp = temp - np.dot(temp, phat)*phat
        self.dipole_moment = temp / norm(temp)  # scale dipole


    def generate_sensor_noise(self, rms_noise=None):
        rms_noise = self.rms_noise if rms_noise is None else rms_noise
        z = np.random.normal(0, 1, self.sensor_locs.shape[0])   # create random noise vector
        noise_scale = rms_noise / np.sqrt(np.var(z))            # get noise scaling factor
        z = noise_scale * z                     # store data for data class
        return z


    #def get_full_magnetic_field(self, lead_field_obj, p = self.dipole_location, q = self.dipole_moment):
    def get_full_magnetic_field(self, lead_field_obj):
        """
        Returns the full magnetic field for class dipole moment and location using as specified lead field object
        :param lead_field_obj: LeadFieldGenerator
        :return: Void
        """
        n_sensors = lead_field_obj.sensors.shape[0]

        p = self.dipole_location
        q = self.dipole_moment

        full_L = lead_field_obj.get_full_lead_field(p)
        dip_B_vec = np.dot(full_L, q)  # this should generate a vector of length (3 * n_sensors)
        dip_B_mat = np.reshape(dip_B_vec, (n_sensors, 3))  # will reshape B field vector to be matrix
        return dip_B_mat


    def get_multidipole_scalar_data(self, Ps, Qs, n_to_avg = 10):
        # randomly generate dipoles somehow maybe pass as an argument
        lfo = self.lead_field_obj
        n_sensors = self.sensor_locs.shape[0]
        rms_noise = self.rms_noise

        # this loop will give us a field in vector form from the randomly drawn dipoles provided.
        B_ambient = np.zeros(lfo.get_full_lead_field(Ps[0,:]).shape[0])
        for i, (p, q) in enumerate(zip(Ps, Qs)):
            temp_L = lfo.get_full_lead_field(p)
            B_temp = np.dot(temp_L, q)
            B_ambient = B_ambient + B_temp

        # add bias field
        B_ambient = B_ambient + np.tile(self.bias_field, (n_sensors,))
        Z = np.zeros((n_sensors, n_to_avg))
        for j in range(n_to_avg):
            Z[:,j] = self.generate_sensor_noise(rms_noise)  # get sensor noise

        z_avg = np.mean(Z,axis=1)
        B_ambient = np.reshape(B_ambient, (n_sensors, 3))
        B_background = norm(B_ambient, axis=1) + z_avg

        B_source = self.get_full_magnetic_field(lfo)
        B_total = norm(B_source + B_ambient, axis=1)
        B_total  = B_total + self.generate_sensor_noise(rms_noise)

        # this returns total field with noise with background field removed
        return B_total - B_background



    def get_vector_sensor_data(self):
        """
        get_vector_sensor_data(self,rms_noise = 0)

        The method returns data generated from vector sensors
        :return: void. Generates noiseless (clean_sensor_data) and noisy data (sensor_data)
        """
        rms_noise = self.rms_noise
        p = self.dipole_location
        q = self.dipole_moment
        L = self.dip_loc.calculate_lead_fields(p)       # generate lead fields
        clean_data = np.dot(L, q)                       # get magnetic fields
        z = self.generate_sensor_noise(rms_noise)       # get sensor noise
        sensor_data = clean_data + z                    # add noise to clean data
        self.clean_sensor_data = clean_data             # store data in class.
        self.sensor_data = sensor_data
        return sensor_data


    def get_scalar_sensor_data(self):
        """
        Method to generator scalar sensor data by subtracting out norm of bias field
        """
        bias_field = self.bias_field
        rms_noise = self.rms_noise
        n_sensors = self.sensor_locs.shape[0]
        dip_B_mat = self.get_full_magnetic_field(self.lead_field_obj)      # call code to generate full B-field
        bias_mat = np.tile(bias_field, (n_sensors,1))   # set bias field at all sensors
        full_B = dip_B_mat + bias_mat                   # add to get total field at each sensor location
        pre_clean = norm(full_B, axis = 1)           # take norm of each row to find mag of field at that point.

        clean_data = pre_clean - norm(bias_field)    # subtract bias field strength to give scalar sensor readings
        z = self.generate_sensor_noise(rms_noise)       # generate noise
        sensor_data = clean_data + z
        self.full_bfield = full_B
        self.clean_sensor_data = clean_data
        self.sensor_data = sensor_data
        return sensor_data

    def get_sensor_data_radial_gradiometry(self):
        temp = self.get_scalar_sensor_data_radial_gradiometry()
        return temp

    def get_scalar_sensor_data_radial_gradiometry(self):
        """
        In this segment, we still need to know the bias field, and will generate full data for all sensors, then
        subtract off the norm of the reading on the corresponding sensors and see how it is.

        :return:
        """

        # extract data settings so we can
        model_type = self.model_type
        full_sensors = self.full_sensors
        full_dirs = self.full_dirs
        bias_field = self.bias_field
        n_sensors = full_sensors.shape[0]
        n_data = int(n_sensors/2)

        # create a new lead field object for the entire sensor array (gradiometers included)
        if self.gradiometry_lead_field_obj is None:
            self.gradiometry_lead_field_obj = LeadFieldGenerator(model_type, full_sensors, full_dirs)

        dip_B_mat = self.get_full_magnetic_field(self.gradiometry_lead_field_obj)      # call code to generate full B-field
        if isinstance(bias_field, (np.ndarray,list,tuple) ):
            bias_mat = np.tile(bias_field, (n_sensors, 1))  # set bias field at all sensors
        else:
            if bias_field == 'radial':
                if 'bias_field_magnitude' in self.settings['sensors'].keys():
                    mag = self.settings['sensors']['bias_field_magnitude']
                else:
                    mag = 1.0e-5
                bias_mat = mag * full_dirs
            else:
                print("Can't recognize bias field type")
        full_B = dip_B_mat + bias_mat                   # add to get total field at each sensor location
        pre_clean = norm(full_B, axis = 1)           # take norm of each row to find mag of field at that point.
        z1 = self.generate_sensor_noise()
        z2 = self.generate_sensor_noise()
        z = np.hstack((z1,z2))
        pre_clean = pre_clean
        clean_data = pre_clean[:n_data] - pre_clean[n_data:]    # subtract corresponding gradiometer norm away
        temp = pre_clean + z
        sensor_data = temp[:n_data] - temp[n_data:]
        self.pre_clean = pre_clean                      # preclean is sensor data for ALL sensors
        self.clean_sensor_data = clean_data             # note that clean is still the diff of gradiometers, not really clean
        self.sensor_data = sensor_data
        self.noise = z
        return sensor_data

    def get_vector_sensor_data_radial_gradiometry(self):
        # NOTE, WE FIX THE ORIENTATION OF EARTHFIELD IN THIS DATA GENERATION ROUTINE WHICH ISN'T THE BEST WAY TO DO THIS
        # MAYBE AT A LATER DATA WE SHOULD AMEND SO ITS NOT HARDCODED. FOR SIMPLICITY, WE LEAVE THIS WAY FOR NOW.
        earth_field = np.array([0, 0, 1e-5])
        model_type = self.model_type
        full_sensors = self.full_sensors
        full_dirs = self.full_dirs
        bias_field = self.bias_field
        n_sensors = full_sensors.shape[0]
        n_data = int(n_sensors / 2)

        # create a new lead field object for the entire sensor array (gradiometers included)
        if self.gradiometry_lead_field_obj is None:
            self.gradiometry_lead_field_obj = LeadFieldGenerator(model_type, full_sensors, full_dirs)

        p = self.dipole_location
        q = self.dipole_moment

        L = self.gradiometry_lead_field_obj.get_lead_field(p)  # get lead field
        dip_B_vec = np.dot(L, q)  # create readings at sensors from dipolar field

        earth_B_vec = np.dot(full_dirs, earth_field)    # find earth field contribution at the sensors.
        pre_clean = dip_B_vec + earth_B_vec             # super-impose earth and dipole field
        z1 = self.generate_sensor_noise()
        z2 = self.generate_sensor_noise()
        z = np.hstack((z1, z2))
        temp = pre_clean + z
        sensor_data = temp[:n_data] - temp[n_data:]
        return sensor_data


    ####################################################################################################################
    ####################################################################################################################
    #######################              PERTURBED DATA GENERATION               #######################################
    ####################################################################################################################
    ####################################################################################################################
    def get_perturbed_vector_sensor_data(self, loc_pert, pol_pert, azi_pert):
        """

        The method returns data generated from vector sensors
        :return: void. Generates noiseless (clean_sensor_data) and noisy data (sensor_data)
        """
        # we need to loop through radial field sensor directions, perturb each by
        model_type = self.model_type
        p = self.dipole_location
        q = self.dipole_moment
        #full_sensors = self.full_sensors
        #full_dirs = self.full_dirs

        true_sensor_directions = self.sensor_dirs
        true_sensor_locs = self.sensor_locs

        # add perturbations to sensor locations
        pert_sensor_locs = true_sensor_locs + loc_pert
        pert_sensor_dirs = np.zeros(true_sensor_directions.shape)
        for i, (theta, phi,true_sensor_dir) in enumerate(zip(pol_pert, azi_pert, true_sensor_directions)):
            # change direction to spherical
            true_dir_spherical = get_spherical_coords(true_sensor_dir[0], true_sensor_dir[1], true_sensor_dir[2])

            # add spherical perturbation to spherical sensor arrangements
            pert_dir_spherical = true_dir_spherical + (np.pi / 180) * np.array([0, theta, phi])

            # transform perturbed sensor directions back to cartesian
            pert_dir_cartesian = get_cartesian_coords(pert_dir_spherical[0],pert_dir_spherical[1],pert_dir_spherical[2])

            # store normalized directions of the sensors
            pert_sensor_dirs[i,:] = pert_dir_cartesian/norm(pert_dir_cartesian)

        # create a lead field object from which we generate data based on perturbed sensors and locations
        lead_field_obj = LeadFieldGenerator(model_type, pert_sensor_locs, pert_sensor_dirs)

        # create lead field for TRUE data at point p
        L = lead_field_obj.get_lead_field(p)

        rms_noise = self.rms_noise
        clean_data = np.dot(L, q)                       # get magnetic fields
        z = self.generate_sensor_noise(rms_noise)       # get sensor noise
        sensor_data = clean_data + z                    # add noise to clean data
        self.clean_sensor_data = clean_data             # store data in class.
        self.sensor_data = sensor_data
        return sensor_data


    def get_perturbed_scalar_sensor_data(self, loc_pert, pol_pert, azi_pert):
        """
        Method to generator scalar sensor data by subtracting out norm of bias field
        """
        model_type = self.model_type
        bias_field = self.bias_field            # extract bias field we believe we are using based on (incorrect) forward model
        rms_noise = self.rms_noise              # get noise level to add to data specified in YAML file
        n_sensors = self.sensor_locs.shape[0]   # determine number of sensors for bias field matrix size
        pert_locs = self.sensor_locs + loc_pert # add location perturbations to model specified sensor locations

        true_dir_spherical = get_spherical_coords(bias_field[0], bias_field[1], bias_field[2])      # get spherical coordinates for sensor directions, that is, the coordinates for bias field
        pert_dir_spherical = true_dir_spherical + (np.pi / 180) * np.array([0, pol_pert, azi_pert]) # add spherical perturbations to the spherical coordinates of bias field
        pert_sensor_dir = get_cartesian_coords(1, pert_dir_spherical[1], pert_dir_spherical[2])     # transform perturbed spherical coordinates back into cartesian with unit norm
        pert_lead_field_obj = LeadFieldGenerator(model_type, pert_locs, pert_sensor_dir)            # create a new lead field with perturbed sensor locations and directions to create our true data

        dip_B_mat = self.get_full_magnetic_field(pert_lead_field_obj)           # use perturbed lead field martrix to generate a full B field from dipole source
        bias_mat = np.tile(norm(bias_field)*pert_sensor_dir, (n_sensors,1))     # create a matrix with full bias field at each sensor location. Note the true bias field is in perturbed direct
        full_B = dip_B_mat + bias_mat               # add bias and dipole field together and sensors
        pre_clean = norm(full_B, axis = 1)          # generate magnitude sensor readings without noise
        clean_data = pre_clean - norm(bias_field)   # subtract out size of bias field from magnitude sensor readings
        z = self.generate_sensor_noise(rms_noise)   # generate noise
        sensor_data = clean_data + z                # add noise to "processed data" with bias field norm removed
        return sensor_data














    ###### THIS IS THE CODE WITH SOME KIND OF ERROR IN IT FOR VARIED_PERTURBATIONS_ON_PHANTOM.PY

    # hopefully this isn't true since I am trying to use it again.

    def get_perturbed_sensor_data_radial_gradiometry(self, loc_pert, pol_pert, azi_pert):
        """
        :param loc_pert: ndarray (sensor location perturbation matrix in meters)
        :param pol_pert: float (polar angle perturbation in degrees)
        :param azi_pert: float (azimuthal angle perturbation in degrees)
        """

        # extract data settings
        model_type = self.model_type
        full_sensors = copy.copy(self.full_sensors + np.tile(loc_pert, (2,1)) )   # full sensors are perturbed
        full_dirs = copy.copy(self.full_dirs)          # PERTURB THESE TOO BY SOME ORIENTATION ANGLE
        bias_field = self.bias_field        # gives bias field type
        n_sensors = full_sensors.shape[0]   # all sensors including gradiometers
        n_data = n_sensors//2           # number of gradiometer pairs

        if isinstance(bias_field, (np.ndarray, list, tuple)):
            # DOUBLE CHECK SCALING IS OK IN HERE
            bfs = get_spherical_coords(bias_field[0], bias_field[1], bias_field[2])

            # add perturbations to polar and azimuthal angle (no change to magnitude so still unit norm)
            pbfs = bfs + np.pi/180 * np.array([0,pol_pert, azi_pert])

            # return cart coords for perturbed orientation unit vector
            pbfc = get_cartesian_coords(pbfs[0], pbfs[1], pbfs[2])

            # repeat sensor orientation for each sensor
            bias_mat = np.tile(pbfc, (n_sensors, 1))

            # this should be the same as matrix above since perturbed bias field cartesian should be unit
            full_dirs = np.tile(pbfc/norm(pbfc), (n_sensors, 1))
        else:
            if bias_field == 'radial':
                # get bias field magnitude from settings or set it to 1e-5
                if 'bias_field_magnitude' in self.settings['sensors'].keys():
                    mag = self.settings['sensors']['bias_field_magnitude']
                else:
                    print("settings['sensors']['bias_field_magnitude'] not specified in config dictionary...using 1e-5 instead")
                    mag = 1.0e-5

                # go through each sensor direction
                for ii, fd in enumerate(full_dirs):
                    # get spherical coords for sensor directions (fd[0] should equal 1)
                    fds = get_spherical_coords(fd[0], fd[1], fd[2])

                    # add perturbation to each direction
                    pfds = fds + np.pi/180 * np.array([0,pol_pert, azi_pert])

                    # recover cart coords of perturbed vector
                    pfdc = get_cartesian_coords(pfds[0], pfds[1], pfds[2])
                    full_dirs[ii, :] = pfdc

                # scale for appropriate bias field strength
                bias_mat = mag * full_dirs
            else:
                print("Can't recognize bias field type")

        # We amended full_sensors and full_dirs above to represent the perturbed and "true" values
        gradiometry_lead_field_obj = LeadFieldGenerator(model_type, full_sensors, full_dirs) # changing full di
        dip_B_mat = self.get_full_magnetic_field(gradiometry_lead_field_obj)

        full_B = dip_B_mat + bias_mat                   # add to get total field at each sensor location
        pre_clean = norm(full_B, axis = 1)           # take norm of each row to find mag of field at that point.
        z1 = self.generate_sensor_noise()
        z2 = self.generate_sensor_noise()
        z = np.hstack((z1, z2))
        pre_clean = pre_clean
        temp = pre_clean + z
        sensor_data = temp[:n_data] - temp[n_data:]
        if np.isnan(sensor_data[0]):
            print('What is happening')
        return sensor_data
