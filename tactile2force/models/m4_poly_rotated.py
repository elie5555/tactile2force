import preprocessing.tactile_to_tip_frame as ttf
import parameters.xela_params as xela_params
from preprocessing.get_polynomial_regressors import get_polynomial_regressors_proj
import numpy as np

class M4:
    def __init__(self):
        self.model_params = None
        self.residuals = None
        self.rank = None
        self.singular_values = None
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
        projected_tactile = np.sum(rotated_tactile, axis=1)

        # Add a column of ones to the data points to account for the bias term
        n = tactile_data.shape[0]
        tactile_data = get_polynomial_regressors_proj(projected_tactile)
        tactile_data = np.hstack((tactile_data.reshape(n, -1), np.ones((n, 1))))

        return tactile_data

    def fit(self, tactile_data, force_data):
        tactile_data_processed = self._process_data(tactile_data)
        force_data
        self.model_params, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(tactile_data_processed, force_data)
        

        #tactile_data_processed = self._process_data(tactile_data)
        # Fit the linear model using least squares
        #self.model_params, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(tactile_data_processed, force_data, rcond=None)

        return self.predict(tactile_data)

    def predict(self, tactile_data):
        projected_tactile = self._process_data(tactile_data)

        # Predict the regressed signal using the fitted model
        regressed_signal = np.dot(projected_tactile, self.model_params)

        return regressed_signal