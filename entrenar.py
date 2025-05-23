#pip install tensorflow
#pip install scikit-learn
#pip install mediapipe
#pip install opencv-python
#pip install pandas
#pip install matplotlib



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


train_df = pd.read_csv('train_data.csv')
valid_df = pd.read_csv('valid_data.csv')
test_df  = pd.read_csv('test_data.csv')

X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values

X_valid = valid_df.drop(columns=['label']).values
y_valid = valid_df['label'].values

X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)



le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_valid_enc = to_categorical(le.transform(y_valid))
y_test_enc  = to_categorical(le.transform(y_test))



le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_valid_enc = to_categorical(le.transform(y_valid))
y_test_enc  = to_categorical(le.transform(y_test))




model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(y_train_enc.shape[1], activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train_enc,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, y_valid_enc)
)

test_loss, test_acc = model.evaluate(X_test, y_test_enc)
print(f"Precisión en test: {test_acc:.2f}")

# Guardar modelo, scaler y encoder
model.save("model_sign_language.keras")

import joblib

# Guardar el codificador de etiquetas
joblib.dump(le, 'label_encoder.jb')
# Guarda el scaler para usarlo en app.py
joblib.dump(scaler, 'scaler.jb')