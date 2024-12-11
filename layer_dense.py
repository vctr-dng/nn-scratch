import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike

SEED = 0
key = random.key(SEED)

class LayerDense:
    weight_type = jnp.float32
    
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights:ArrayLike = 0.1 * random.normal(key, (n_inputs, n_neurons), dtype=self.weight_type)
        self.biases:ArrayLike = jnp.zeros((1, n_neurons), dtype=self.weight_type)
        
    def forward(self, inputs:ArrayLike) -> ArrayLike:
        self.output = jnp.dot(inputs, self.weights) + self.biases


if __name__ == "__main__":
    # Create a dense layer with 3 input features and 2 output features
    layer1 = LayerDense(3, 2)
    layer2 = LayerDense(2, 3)
    
    # Create a random input vector
    X = jnp.array([[1.0, 2.0, 3.0]])
    
    # Perform a forward pass of our training data through this layer
    layer1.forward(X)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)