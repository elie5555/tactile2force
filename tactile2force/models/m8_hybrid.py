import m3_parameter_free_linear as m3
import m6_cnn_2 as m6

class hybrid():
    def __init__(self) -> None:
        self.m3 = m3.M3()
        self.m6 = m6.SimpleCNN((6,6))

    def train_network(self, x_train, y_train, x_val, num_epochs=120, learning_rate=0.00025):
        m3_fit = self.m3.fit(x_train, y_train)
        val_fit = self.m3.predict(x_val)
        train_error = y_train - m3_fit
        val_error = x_val - val_fit