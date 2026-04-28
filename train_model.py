import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib   # ← added for scaler save
import os
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE       = 50
WINDOW_SECONDS    = 2
WINDOW_SIZE       = SAMPLE_RATE * WINDOW_SECONDS   # 100
STEP_SIZE         = 25
N_FEATURES        = 6
N_CLASSES         = 3
EPOCHS            = 40
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-3
RANDOM_SEED       = 42
MODEL_PATH        = "fall_detection_cnn1d.keras"
TFLITE_PATH       = "fall_detection_cnn1d.tflite"
SCALER_PATH       = "scaler.pkl"          # ← added

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
CLASS_NAMES = {0: "Normal ADL", 1: "Fall", 2: "Lying still"}

def _add_noise(signal, std=0.05):
    return signal + np.random.normal(0, std, signal.shape)

def generate_normal_segment(length):
    t  = np.linspace(0, length / SAMPLE_RATE, length)
    ax = _add_noise(0.3  * np.sin(2 * np.pi * 1.2 * t))
    ay = _add_noise(0.2  * np.sin(2 * np.pi * 0.8 * t + 0.5))
    az = _add_noise(np.ones(length) * 9.81 + 0.1 * np.sin(2 * np.pi * 2 * t))
    gx = _add_noise(15   * np.sin(2 * np.pi * 1.0 * t))   # ← realistic deg/s
    gy = _add_noise(10   * np.sin(2 * np.pi * 0.6 * t + 1.0))
    gz = _add_noise(8    * np.cos(2 * np.pi * 0.9 * t))
    return np.column_stack([ax, ay, az, gx, gy, gz])

def generate_fall_segment(length):
    seg = np.zeros((length, N_FEATURES))
    impact = length // 3
    seg[:impact, 2]  = np.linspace(9.81, -4.0, impact) + np.random.normal(0, 0.3, impact)
    seg[:impact, 0]  = np.random.normal(0.5, 0.4, impact)
    seg[:impact, 3:] = np.random.normal(0, 80, (impact, 3))   # ← fall gyro spike deg/s
    spike_len = max(5, length // 10)
    spike = np.concatenate([np.linspace(0, 25, spike_len // 2),
                             np.linspace(25, 0, spike_len - spike_len // 2)])
    end_spike = min(impact + spike_len, length)
    real_len  = end_spike - impact
    seg[impact:end_spike, 0]  = spike[:real_len]
    seg[impact:end_spike, 2]  = -spike[:real_len] * 0.6
    seg[impact:end_spike, 3:] = np.random.normal(0, 150, (real_len, 3))  # ← impact spike
    post = end_spike
    if post < length:
        seg[post:, 2]  = _add_noise(np.ones(length - post) * 9.81, std=0.15)
        seg[post:, :2] = np.random.normal(0, 0.05, (length - post, 2))
        seg[post:, 3:] = np.random.normal(0, 2, (length - post, 3))
    return seg

def generate_lying_segment(length):
    ax = _add_noise(np.zeros(length), std=0.08)
    ay = _add_noise(np.zeros(length), std=0.08)
    az = _add_noise(np.ones(length) * 9.81, std=0.15)
    gx = _add_noise(np.zeros(length), std=1.0)
    gy = _add_noise(np.zeros(length), std=1.0)
    gz = _add_noise(np.zeros(length), std=0.8)
    return np.column_stack([ax, ay, az, gx, gy, gz])

def generate_raw_dataset(n_normal=600, n_fall=250, n_lying=350, seg_length=300):
    segments, labels = [], []
    for _ in range(n_normal):
        segments.append(generate_normal_segment(seg_length)); labels.append(0)
    for _ in range(n_fall):
        segments.append(generate_fall_segment(seg_length));   labels.append(1)
    for _ in range(n_lying):
        segments.append(generate_lying_segment(seg_length));  labels.append(2)
    return segments, labels

def extract_windows(segments, labels, window_size=WINDOW_SIZE, step=STEP_SIZE):
    X, y = [], []
    for seg, label in zip(segments, labels):
        for start in range(0, len(seg) - window_size + 1, step):
            X.append(seg[start:start + window_size])
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def preprocess(X_train, X_val, X_test):
    n_train, wlen, nfeat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, nfeat))
    X_train = scaler.transform(X_train.reshape(-1, nfeat)).reshape(n_train, wlen, nfeat)
    X_val   = scaler.transform(X_val.reshape(-1, nfeat)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, nfeat)).reshape(X_test.shape)
    joblib.dump(scaler, SCALER_PATH)          # ← save scaler
    print(f"      Scaler saved → {SCALER_PATH}")
    return X_train, X_val, X_test, scaler

def build_model(window_size=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES):
    inp = keras.Input(shape=(window_size, n_features), name="imu_input")
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=out, name="FallDetect_CNN1D")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=0)
    ]
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_list, verbose=1)

def evaluate_model(model, X_test, y_test):
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()), zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':16s}", "  ".join(f"{n:12s}" for n in CLASS_NAMES.values()))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:16s}", "  ".join(f"{v:12d}" for v in row))

def convert_to_tflite(model):
    print("\n" + "="*60)
    print("TFLite CONVERSION")
    print("="*60)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"  Float32 TFLite saved → {TFLITE_PATH}  ({os.path.getsize(TFLITE_PATH)/1024:.1f} KB)")
    return tflite_model

def run_tflite_inference(X_test, y_test, n_samples=10):
    print("\n" + "="*60)
    print("TFLite SANITY CHECK")
    print("="*60)
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    idxs    = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    correct = 0
    for i in idxs:
        interp.set_tensor(in_det["index"], X_test[i:i+1].astype(np.float32))
        interp.invoke()
        pred = np.argmax(interp.get_tensor(out_det["index"]))
        true = y_test[i]
        print(f"  [{'✓' if pred==true else '✗'}] True: {CLASS_NAMES[true]:14s} | Pred: {CLASS_NAMES[pred]}")
        correct += (pred == true)
    print(f"\n  Quick-check accuracy: {correct}/{len(idxs)}")

def main():
    print("="*60)
    print("FALL DETECTION — CNN-1D PIPELINE")
    print("="*60)

    print("\n[1/6] Generating synthetic dataset …")
    segments, labels = generate_raw_dataset()
    X, y = extract_windows(segments, labels)
    print(f"      Windows: {X.shape}  |  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n[2/6] Splitting …")
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.18, stratify=y_tmp, random_state=RANDOM_SEED)
    print(f"      Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

    print("\n[3/6] Normalising …")
    X_train, X_val, X_test, scaler = preprocess(X_train, X_val, X_test)

    print("\n[4/6] Building model …")
    model = build_model()
    model.summary()

    print("\n[5/6] Training …")
    train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)

    print("\n[6/6] Exporting …")
    convert_to_tflite(model)
    run_tflite_inference(X_test, y_test)

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"  Keras  : {MODEL_PATH}")
    print(f"  TFLite : {TFLITE_PATH}")
    print(f"  Scaler : {SCALER_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()