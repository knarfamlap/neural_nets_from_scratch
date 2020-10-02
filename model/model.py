from utils.loss import mse, mse_prime

class Model:
    def __init__(self):
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data): 
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        return result

    def fit(self, X_train, y_train, epochs, learning_rate):
        samples = len(X_train)

        for i in range(epochs):
            loss = 0 
            for j in range(samples):
                output = X_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output) 

                loss += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output) 

                for layer in reversed(self.layers): 
                    error = layer.backward_propagation(error, learning_rate)

            loss /= samples
            print('Epoch %d, Error=%f' % (i+1, loss))


            
