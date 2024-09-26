import preprocessing.tactile_to_tip_frame as ttf
import parameters.xela_params as xela_params
import numpy as np

class M0:
    def __init__(self):
        self.model_params_x = None
        self.model_params_y = None
        self.model_params_z = None
        self.residuals_x = None
        self.residuals_y = None
        self.residuals_z = None
        self.rank_x = None
        self.rank_y = None
        self.rank_z = None
        self.singular_values_x = None
        self.singular_values_y = None
        self.singular_values_z = None
        self.transforms = xela_params.tip_tf()

    def _to_numpy(data):
        if isinstance(data, np.ndarray):
            # Proceed with processing the numpy array
            print("Processing NumPy array")
        else:
            data = data.numpy()

    def _process_data(self, tactile_data):
        # Rotate the tactile data to the fingertip frame
        if tactile_data.shape[1] == xela_params.N_TAXEL_TIP:
            rotation_matrix = ttf.rpy_to_rotation_matrix(self.transforms.get_rpy_array(), 
                                                        self.transforms.get_sensor_o_to_fingertip(), self.transforms.get_measurement_to_taxel())
            rotated_tactile = ttf.rotate_tactile(tactile_data, rotation_matrix)
        elif tactile_data.shape[1] == xela_params.N_TAXEL_PHAL:
            rotation_matrix =xela_params.phal_rot
            rotated_tactile = ttf.rotate_phal_patch(tactile_data, rotation_matrix)

        # Sum the tactile data along the x, y, and z dimensions
        x_tactile = np.sum(rotated_tactile[:, :, 0], axis=1)
        y_tactile = np.sum(rotated_tactile[:, :, 1], axis=1)
        z_tactile = np.sum(rotated_tactile[:, :, 2], axis=1)

        n = rotated_tactile.shape[0]
        x_tactile = x_tactile.reshape((n, 1))
        y_tactile = y_tactile.reshape((n, 1))
        z_tactile = z_tactile.reshape((n, 1))

        # Add a column of ones to the data points to account for the bias term
        #n = rotated_tactile.shape[0]
        #x_tactile = np.hstack((x_tactile.reshape((n, 1)), np.ones((n, 1))))
        #y_tactile = np.hstack((y_tactile.reshape((n, 1)), np.ones((n, 1))))
        #z_tactile = np.hstack((z_tactile.reshape((n, 1)), np.ones((n, 1))))

        return x_tactile, y_tactile, z_tactile

    def fit(self, tactile_data, force_data):
        x_tactile, y_tactile, z_tactile = self._process_data(tactile_data)

        # Fit the linear model using least squares
        #print(x_tactile.shape)
        #print(y_tactile.shape)
        #print(z_tactile.shape)

        #print(np.linalg.cond(np.dot(x_tactile.T, x_tactile)))
        #print(np.linalg.cond(np.dot(y_tactile.T, y_tactile)))
        #print(np.linalg.cond(np.dot(z_tactile.T, z_tactile)))

        self.model_params_x, self.residuals_x, self.rank_x, self.singular_values_x = np.linalg.lstsq(x_tactile, force_data[:,0], rcond=None)
        self.model_params_y, self.residuals_y, self.rank_y, self.singular_values_y = np.linalg.lstsq(y_tactile, force_data[:,1], rcond=None)
        self.model_params_z, self.residuals_z, self.rank_z, self.singular_values_z = np.linalg.lstsq(z_tactile, force_data[:,2], rcond=None)

        return self.predict(tactile_data)

    def predict(self, tactile_data):
        x_tactile, y_tactile, z_tactile = self._process_data(tactile_data)

        # Predict the regressed signal using the fitted model
        regressed_signal_x = np.dot(x_tactile, self.model_params_x)
        regressed_signal_y = np.dot(y_tactile, self.model_params_y)
        regressed_signal_z = np.dot(z_tactile, self.model_params_z)

        # Adds a new axis to the regressed signals for concatenation
        n = regressed_signal_x.shape[0]
        regressed_signal_x = regressed_signal_x.reshape((n, 1))
        regressed_signal_y = regressed_signal_y.reshape((n, 1))
        regressed_signal_z = regressed_signal_z.reshape((n, 1))

        return np.hstack((regressed_signal_x, regressed_signal_y, regressed_signal_z))