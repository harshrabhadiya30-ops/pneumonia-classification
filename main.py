"""
Assignment 2: Image-Based Disease Classification (Chest X-ray: NORMAL vs PNEUMONIA)

Run from the folder that contains `archive/chest_xray/{train,val,test}`:
    python main.py

Needs: tensorflow, scikit-learn, pandas, matplotlib, pillow, joblib

Pipeline
  1. Data understanding  -> class balance, sample images, image sizes
  2. Data preparation    -> own stratified train/val split, resize, class weights, augmentation
  3. Models              -> Baseline (Logistic Regression)
                            CNN from scratch
                            Transfer learning (MobileNetV2, 2-phase fine-tuning)
  4. Evaluation          -> accuracy, precision, recall, F1, AUC, confusion matrices, ROC
  5. Comparison          -> table + bar chart
  6. Explainability      -> Grad-CAM heatmaps (bonus)
  7. Demo                -> predicted class + confidence for single images
"""
import os
import json
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import joblib
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ----------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------
DATA_DIR = Path("archive/chest_xray")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

CLASSES = ["NORMAL", "PNEUMONIA"]          # label 0 = NORMAL, label 1 = PNEUMONIA
IMG = 160                                  # all deep models use 160x160 RGB
BATCH = 32
BASELINE_SIZE = 64                         # baseline uses 64x64 grayscale pixels

# SMOKE=1 is only for a quick pipeline check (tiny epochs, no pretrained download)
SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS_CNN = 2 if SMOKE else 15
EPOCHS_TL_HEAD = 1 if SMOKE else 5
EPOCHS_TL_FINETUNE = 1 if SMOKE else 6
PRETRAINED = None if SMOKE else "imagenet"


def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 1. Data understanding
# ----------------------------------------------------------------------------
def collect(split):
    rows = []
    for cls in CLASSES:
        folder = DATA_DIR / split / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        for p in glob.glob(str(folder / "*")):
            if p.lower().endswith((".jpeg", ".jpg", ".png")):
                rows.append((p, cls, split))
    return pd.DataFrame(rows, columns=["path", "label_name", "orig_split"])


# The official 'val' folder holds only 16 images: far too few for reliable
# early stopping, so we pool train+val and create our own validation split.
df_dev = pd.concat([collect("train"), collect("val")], ignore_index=True)
df_test = collect("test")
for d in (df_dev, df_test):
    d["label"] = (d["label_name"] == "PNEUMONIA").astype(int)

print(f"Train+val pool: {len(df_dev)} images | Test: {len(df_test)} images")
print("Dev class balance:\n", df_dev["label_name"].value_counts())
print("Test class balance:\n", df_test["label_name"].value_counts())

# class distribution chart
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, (title, d) in zip(ax, [("Train + Val pool", df_dev), ("Test", df_test)]):
    counts = d["label_name"].value_counts().reindex(CLASSES)
    a.bar(counts.index, counts.values, color=["tab:blue", "tab:red"])
    for i, v in enumerate(counts.values):
        a.text(i, v, str(v), ha="center", va="bottom")
    a.set_title(title)
save_fig("class_distribution.png")

# sample images
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for r, cls in enumerate(CLASSES):
    sample = df_dev[df_dev["label_name"] == cls].sample(5, random_state=SEED)
    for c, p in enumerate(sample["path"]):
        axes[r, c].imshow(Image.open(p).convert("L"), cmap="gray")
        axes[r, c].set_title(cls)
        axes[r, c].axis("off")
save_fig("sample_images.png")

# image size variation
sizes = [Image.open(p).size for p in df_dev["path"].sample(min(300, len(df_dev)), random_state=SEED)]
w, h = zip(*sizes)
print(f"Image width  min/mean/max: {min(w)}/{int(np.mean(w))}/{max(w)}")
print(f"Image height min/mean/max: {min(h)}/{int(np.mean(h))}/{max(h)}")

# ----------------------------------------------------------------------------
# 2. Data preparation
# ----------------------------------------------------------------------------
df_train, df_val = train_test_split(
    df_dev, test_size=0.2, stratify=df_dev["label"], random_state=SEED
)
print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# class weights: rarer class (NORMAL) counts more during training
n = len(df_train)
n0, n1 = (df_train["label"] == 0).sum(), (df_train["label"] == 1).sum()
class_weight = {0: n / (2 * n0), 1: n / (2 * n1)}
print("Class weights:", {k: round(v, 2) for k, v in class_weight.items()})


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG, IMG))
    return img, label


def make_ds(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("float32"))
    )
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(df_train, shuffle=True)
val_ds = make_ds(df_val)
test_ds = make_ds(df_test)            # NOT shuffled: predictions stay aligned with df_test
y_test = df_test["label"].values

# Augmentation (active only while training). No flips: chest X-rays have a fixed
# orientation, so mirrored images would be unrealistic.
augment = keras.Sequential(
    [layers.RandomRotation(0.05), layers.RandomZoom(0.1), layers.RandomContrast(0.1)],
    name="augmentation",
)

# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
results, all_probs = [], {}


def evaluate(name, slug, y_true, prob, thr=0.5):
    y_pred = (prob >= thr).astype(int)
    print(f"\n===== {name}: TEST RESULTS =====")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES,
                                digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix: {name}")
    save_fig(f"confusion_matrix_{slug}.png")
    tn, fp, fn, tp = cm.ravel()
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, prob),
        "FalseNegatives": int(fn),
        "FalsePositives": int(fp),
    }
    results.append(row)
    all_probs[name] = prob
    with open(OUTPUT_DIR / f"metrics_{slug}.json", "w") as f:
        json.dump({k: (v if isinstance(v, str) else float(v)) for k, v in row.items()}, f, indent=2)


def plot_history(histories, title, fname):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    pd.DataFrame(merged).to_csv(OUTPUT_DIR / fname.replace(".png", ".csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(merged["loss"], label="train")
    ax[0].plot(merged["val_loss"], label="validation")
    ax[0].set_title(f"{title}: loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(merged["accuracy"], label="train")
    ax[1].plot(merged["val_accuracy"], label="validation")
    ax[1].set_title(f"{title}: accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    save_fig(fname)


# ----------------------------------------------------------------------------
# 3a. Baseline: Logistic Regression on raw pixels
# ----------------------------------------------------------------------------
def load_flat(paths, size=BASELINE_SIZE):
    X = []
    for p in paths:
        im = Image.open(p).convert("L").resize((size, size))
        X.append(np.asarray(im, dtype="float32").ravel() / 255.0)
    return np.array(X)


print("\nTraining baseline (Logistic Regression)...")
X_train, X_val, X_test = (load_flat(d["path"]) for d in (df_train, df_val, df_test))
baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
baseline.fit(X_train, df_train["label"])
print("Baseline validation accuracy:", round(baseline.score(X_val, df_val["label"]), 3))
joblib.dump(baseline, MODEL_DIR / "baseline_logreg.joblib")
evaluate("Baseline: Logistic Regression", "baseline", y_test, baseline.predict_proba(X_test)[:, 1])

# ----------------------------------------------------------------------------
# 3b. Advanced model 1: CNN from scratch
# ----------------------------------------------------------------------------
METRICS = ["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]


def build_cnn():
    inp = keras.Input((IMG, IMG, 3))
    x = augment(inp)
    x = layers.Rescaling(1.0 / 255)(x)
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="simple_cnn")


print("\nTraining CNN from scratch...")
cnn = build_cnn()
cnn.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=METRICS)
cnn.summary()
h_cnn = cnn.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_CNN, class_weight=class_weight,
    callbacks=[
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_cnn.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ],
)
cnn.save(MODEL_DIR / "cnn_final.keras")
plot_history([h_cnn], "CNN from scratch", "learning_curves_cnn.png")
evaluate("CNN from scratch", "cnn", y_test, cnn.predict(test_ds).ravel())

# ----------------------------------------------------------------------------
# 3c. Advanced model 2: Transfer learning (MobileNetV2)
# ----------------------------------------------------------------------------
print("\nTraining transfer-learning model (MobileNetV2)...")
base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights=PRETRAINED)
base.trainable = False

gap = layers.GlobalAveragePooling2D()
drop = layers.Dropout(0.3)
head = layers.Dense(1, activation="sigmoid")

inp = keras.Input((IMG, IMG, 3))
x = augment(inp)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)      # MobileNetV2 expects pixels in [-1, 1]
x = base(x, training=False)
out = head(drop(gap(x)))
tl = keras.Model(inp, out, name="mobilenetv2_transfer")

# Phase 1: train only the new classification head
tl.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=METRICS)
h1 = tl.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_TL_HEAD, class_weight=class_weight)

# Phase 2: fine-tune the last 30 layers of the pretrained network, tiny learning rate
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
tl.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=METRICS)
h2 = tl.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS_TL_FINETUNE, class_weight=class_weight,
    callbacks=[
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "best_transfer.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ],
)
tl.save(MODEL_DIR / "transfer_final.keras")
plot_history([h1, h2], "MobileNetV2 transfer", "learning_curves_transfer.png")
evaluate("Transfer learning (MobileNetV2)", "transfer", y_test, tl.predict(test_ds).ravel())

# ----------------------------------------------------------------------------
# 4/5. Comparison
# ----------------------------------------------------------------------------
res = pd.DataFrame(results).set_index("Model")
res.round(4).to_csv(OUTPUT_DIR / "model_comparison.csv")
print("\n===== MODEL COMPARISON (test set) =====")
print(res.round(3).to_string())

res[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(kind="bar", figsize=(10, 4.5), ylim=(0, 1.05))
plt.xticks(rotation=15, ha="right")
plt.title("Model comparison on the test set")
plt.legend(loc="lower right")
save_fig("model_comparison.png")

plt.figure(figsize=(5.5, 5))
for name, prob in all_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate (recall)")
plt.title("ROC curves (test set)"); plt.legend(fontsize=7)
save_fig("roc_curves.png")

# ----------------------------------------------------------------------------
# 6. Explainability: Grad-CAM on the transfer model (bonus)
# ----------------------------------------------------------------------------
feat_model = keras.Model(base.input, base.get_layer("out_relu").output)


def grad_cam(img):
    """img: (IMG, IMG, 3) float array in 0..255. Returns (heatmap 0..1, pneumonia prob)."""
    x = tf.expand_dims(tf.cast(img, tf.float32), 0) / 127.5 - 1.0
    with tf.GradientTape() as tape:
        conv = feat_model(x, training=False)
        tape.watch(conv)
        prob = head(gap(conv))
    grads = tape.gradient(prob, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))                       # importance of each channel
    cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG, IMG))[..., 0].numpy()
    return cam, float(prob[0][0])


pick = pd.concat([
    df_test[df_test["label"] == 0].sample(3, random_state=SEED),
    df_test[df_test["label"] == 1].sample(3, random_state=SEED),
])
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for i, (_, row) in enumerate(pick.iterrows()):
    img, _ = load_image(row["path"], 0.0)
    cam, p = grad_cam(img.numpy())
    axes[0, i].imshow(img.numpy().astype("uint8")); axes[0, i].axis("off")
    axes[0, i].set_title(f"True: {row['label_name']}", fontsize=9)
    axes[1, i].imshow(img.numpy().astype("uint8"))
    axes[1, i].imshow(cam, cmap="jet", alpha=0.45); axes[1, i].axis("off")
    axes[1, i].set_title(f"P(pneumonia)={p:.2f}", fontsize=9)
save_fig("gradcam_examples.png")

# ----------------------------------------------------------------------------
# 7. Per-image predictions with confidence (required output)
# ----------------------------------------------------------------------------
best_name = res.loc[[m for m in res.index if "Baseline" not in m], "F1"].idxmax()
best_prob = all_probs[best_name]
pred = pd.DataFrame({
    "path": df_test["path"].values,
    "actual": df_test["label_name"].values,
    "predicted": np.where(best_prob >= 0.5, "PNEUMONIA", "NORMAL"),
    "confidence": np.where(best_prob >= 0.5, best_prob, 1 - best_prob),
    "pneumonia_score": best_prob,
})
for name, prob in all_probs.items():
    pred[f"score_{name}"] = prob
pred.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
print(f"\nBest deep model by F1: {best_name}")
print(pred[["actual", "predicted", "confidence"]].head(8).to_string())


def predict_image(model, path, thr=0.5):
    """Predict class + confidence for ONE image file with a trained Keras model."""
    img, _ = load_image(path, 0.0)
    p = float(model.predict(tf.expand_dims(img, 0), verbose=0)[0][0])
    return CLASSES[int(p >= thr)], (p if p >= thr else 1 - p)


print("\nExample single-image predictions (transfer model):")
for path in df_test["path"].sample(5, random_state=1):
    label, conf = predict_image(tl, path)
    print(f"  {Path(path).parent.name:10s} -> {label:10s} ({conf:.1%})")

print("\nFinished. Models in models/, figures + tables in outputs/.")
