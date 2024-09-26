import preprocessing.tactile_to_tip_frame as ttf
import parameters.xela_params as xela_params
import numpy as np

class M2:
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
        #print("after summing tactile data along x, y, z dimensions, shape is", projected_tactile.shape)

        # Add a column of ones to the data points to account for the bias term
        n = projected_tactile.shape[0]
        projected_tactile = np.hstack((projected_tactile, np.ones((n, 1))))

        return projected_tactile

    def fit(self, tactile_data, force_data):
        projected_tactile = self._process_data(tactile_data)

        # Fit the linear model using least squares
        self.model_params, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(projected_tactile, force_data, rcond=None)

        return self.predict(tactile_data)

    def predict(self, tactile_data):
        projected_tactile = self._process_data(tactile_data)

        # Predict the regressed signal using the fitted model
        regressed_signal = np.dot(projected_tactile, self.model_params)

        return regressed_signal