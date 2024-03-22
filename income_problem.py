import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample data (you can replace this with your own dataset)
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'education': ['Bachelors', 'Masters', 'Bachelors', 'Masters', 'Doctorate', 'Doctorate', 'Bachelors', 'Masters'],
    'hours_per_week': [40, 50, 45, 55, 60, 35, 50, 45],
    'income': ['<=50k', '>50k', '<=50k', '>50k', '>50k', '<=50k', '<=50k', '>50k']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Encode categorical columns
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])

# Map income to binary labels
df['income'] = df['income'].map({'<=50k': 0, '>50k': 1})

# Split features and target
X = df.drop('income', axis=1)
y = df['income']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

