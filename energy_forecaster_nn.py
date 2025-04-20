import numpy as np
import tensorflow as tf

class EnergyForecasterNN:
    def __init__(self, input_features, n_hidden=[64, 64], seed=42):
        self.input_features = input_features
        self.input_dim = len(input_features)
        self.output_dim = 1  # Single value: energy production
        self.n_hidden = n_hidden
        self.seed = seed
        self._build_tf_model()

    def _custom_activation(self, x, alpha, beta):
        return (beta + (1 - beta) * tf.sigmoid(alpha * x)) * x

    def _build_tf_model(self):
        tf.random.set_seed(self.seed)
        inputs = tf.keras.Input(shape=(self.input_dim,), name="inputs")
        x = inputs

        self.alpha_params = []
        self.beta_params = []

        for i, hidden_size in enumerate(self.n_hidden):
            x = tf.keras.layers.Dense(hidden_size, name=f"dense_{i}")(x)

            # alpha and beta as trainable scalars
            alpha = tf.Variable(tf.random.uniform([1], 0.5, 1.5), name=f"alpha_{i}")
            beta = tf.Variable(tf.random.uniform([1], 0.1, 0.9), name=f"beta_{i}")
            self.alpha_params.append(alpha)
            self.beta_params.append(beta)

            # wrap into custom activation
            x = tf.keras.layers.Lambda(
                lambda x_, alpha=alpha, beta=beta: self._custom_activation(x_, alpha, beta),
                name=f"custom_activation_{i}"
            )(x)

        output = tf.keras.layers.Dense(1, name="output")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, validation_split=0.2, epochs=1000, batch_size=32, patience=30):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def summary(self):
        self.model.summary()

# Example usage
# features = ['diffuse_radiation', 'direct_radiation', 'temperature', 'high_cloud_cover']
# model = EnergyForecasterNN(input_features=features)
# model.summary()
# model.train(X_train, y_train)
# predictions = model.predict(X_test)
