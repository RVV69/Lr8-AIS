import tensorflow as tf
import numpy as np

# Генерація простих даних
X = np.random.rand(1000, 1)
y = 3 * X + 2 + np.random.normal(0, 0.1, (1000, 1))  # Лінійна залежність з шумом

# Оголошуємо модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=1)
])

# Визначаємо оптимізатор з learning rate, який зменшується
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Визначаємо функцію втрат
loss_fn = tf.keras.losses.MeanSquaredError()

# Навчання моделі
epochs = 20000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Прямий прохід
        y_pred = model(X, training=True)
        loss = loss_fn(y, y_pred)

    # Обчислення градієнтів
    grads = tape.gradient(loss, model.trainable_variables)

    # Оновлення ваг
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 100 == 0:
        print(
            f"Епоха {epoch}: Помилка = {loss.numpy():.6f}, k = {model.layers[0].weights[0].numpy()[0][0]:.4f}, b = {model.layers[0].weights[1].numpy()[0]:.4f}")

# Після завершення навчання
print("Навчання завершено.")
print(
    f"Остаточні значення: k = {model.layers[0].weights[0].numpy()[0][0]:.4f}, b = {model.layers[0].weights[1].numpy()[0]:.4f}")
