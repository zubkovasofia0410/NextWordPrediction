from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

model = load_model("models/saved_model.keras")

plot_model(
    model,
    to_file="img/lstm_architecture.png",
    show_shapes=True,
    show_layer_activations=True,
    show_trainable=False,
    show_dtype=False,
    show_layer_names=True,
    expand_nested=True,
    dpi=200
)

model.summary()