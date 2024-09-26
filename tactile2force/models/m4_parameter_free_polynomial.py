from preprocessing.get_polynomial_regressors import get_polynomial_regressors
import numpy as np

class M4:
    def __init__(self):
        self.model_params = None
        self.residuals = None
        self.rank = None
        self.singular_values = None

    def _to_numpy(data):
        if isinstance(data, np.ndarray):
            # Proceed with processing the numpy array
            print("Processing NumPy array")
        else:
            data = data.numpy()

    def _process_data(self, tactile_data):
        # Add a column of ones to the data points to account for the bias term
        n = tactile_data.shape[0]
        tactile_data = get_polynomial_regressors(tactile_data)
        tactile_data = np.hstack((tactile_data.reshape(n, -1), np.ones((n, 1))))

        return tactile_data

    def fit(self, tactile_data, force_data):
        tactile_data_processed = self._process_data(tactile_data)
        A = tactile_data_processed
        n_col = A.shape[1]
        lamb = 15000
        y = force_data
        self.model_params, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))
        

        #tactile_data_processed = self._process_data(tactile_data)
        # Fit the linear model using least squares
        #self.model_params, self.residuals, self.rank, self.singular_values = np.linalg.lstsq(tactile_data_processed, force_data, rcond=None)

        return self.predict(tactile_data)

    def predict(self, tactile_data):
        projected_tactile = self._process_data(tactile_data)

        # Predict the regressed signal using the fitted model
        regressed_signal = np.dot(projected_tactile, self.model_params)

        return regressed_signal