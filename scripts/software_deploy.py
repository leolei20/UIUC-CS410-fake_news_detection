

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


# =========================================================
# Helper: base directory (works for .py and PyInstaller exe)
# =========================================================

def get_base_dir() -> Path:
    """
    Returns the folder where this script / exe lives.
    Works both for plain Python and PyInstaller-built exe.
    """
    return Path(getattr(sys, "_MEIPASS", Path(sys.argv[0]).parent)).resolve()


# =========================================================
# Shared text cleaning (roughly matches your preprocessing)
# =========================================================

def simple_clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# Baseline (TF-IDF + Logistic Regression) predictor
# =========================================================

class BaselinePredictor:
    def __init__(self):
        base_dir = get_base_dir()
        outputs_dir = base_dir.parent / "outputs"
        vectorizer_dir = outputs_dir / "vectorizers"
        model_dir = outputs_dir / "models"

        print("[Baseline] Loading vectorizers and model...")
        self.tfidf_body = joblib.load(vectorizer_dir / "tfidf_body.pkl")
        self.tfidf_title = joblib.load(vectorizer_dir / "tfidf_title.pkl")
        self.scaler = joblib.load(model_dir.parent / "vectorizers" / "style_scaler.pkl") \
            if (model_dir.parent / "vectorizers" / "style_scaler.pkl").exists() \
            else joblib.load(vectorizer_dir / "style_scaler.pkl")
        self.model = joblib.load(model_dir / "baseline_model.pkl")
        print("[Baseline] Loaded.")

    def extract_style_features(self, df: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame()
        text = df["text"].astype(str)
        title = df["title"].astype(str)

        features["text_length"] = text.str.len()
        features["word_count"] = text.str.split().str.len()
        features["title_word_count"] = title.str.split().str.len()

        features["punctuation_ratio"] = text.str.count(r"[!?]") / features["text_length"]
        features["uppercase_ratio"] = text.str.count(r"[A-Z]") / features["text_length"]
        features["digit_ratio"] = text.str.count(r"\d") / features["text_length"]
        features["avg_word_length"] = features["text_length"] / features["word_count"]

        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        return self.scaler.transform(features)

    def predict(self, title: str, body: str):
        text = body
        clean_body = simple_clean_text(text)

        df = pd.DataFrame([{
            "clean_text": clean_body,
            "title": title,
            "text": text,
        }])

        X_body = self.tfidf_body.transform(df["clean_text"])
        X_title = self.tfidf_title.transform(df["title"].fillna(""))

        X_style = self.extract_style_features(df)
        X_final = hstack([X_body, X_title, csr_matrix(X_style)])

        proba_fake = self.model.predict_proba(X_final)[0, 1]
        pred_label = int(self.model.predict(X_final)[0])
        return pred_label, float(proba_fake)


# =========================================================
# DistilBERT predictor
# =========================================================

class BertPredictor:
    def __init__(self, max_length: int = 256):
        base_dir = get_base_dir()
        outputs_dir = base_dir.parent / "outputs"
        model_dir = outputs_dir / "models" / "distilbert"

        print("[BERT] Loading tokenizer and model...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        print(f"[BERT] Using device: {self.device}")
        print("[BERT] Loaded.")

    def predict(self, title: str, body: str):
        text = (title or "") + " " + (body or "")
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            proba_fake = float(probs[1].cpu().item())
            pred_label = int(torch.argmax(probs).cpu().item())

        return pred_label, proba_fake


# =========================================================
# Tkinter GUI
# =========================================================

class FakeNewsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detector")

        # Pre-load both models once at startup
        self.result_var = tk.StringVar(value="Loading models...")
        self.root.update_idletasks()

        try:
            self.baseline_model = BaselinePredictor()
        except Exception as e:
            self.baseline_model = None
            print(f"[Error] Baseline model load failed: {e}")

        try:
            self.bert_model = BertPredictor()
        except Exception as e:
            self.bert_model = None
            print(f"[Error] BERT model load failed: {e}")

        # ---------- Layout ----------
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Title
        ttk.Label(main_frame, text="Title:").grid(row=0, column=0, sticky="w")
        self.title_entry = ttk.Entry(main_frame, width=60)
        self.title_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)

        # Body
        ttk.Label(main_frame, text="Body:").grid(row=1, column=0, sticky="nw")
        self.body_text = scrolledtext.ScrolledText(
            main_frame, width=60, height=10, wrap=tk.WORD
        )
        self.body_text.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=5)

        # Model dropdown
        ttk.Label(main_frame, text="Model:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        model_values = []
        if self.bert_model is not None:
            model_values.append("DistilBERT")
        if self.baseline_model is not None:
            model_values.append("Baseline")

        if not model_values:
            model_values = ["(No model loaded)"]

        self.model_var = tk.StringVar(
            value=model_values[0] if model_values else "(No model loaded)"
        )
        self.model_combo = ttk.Combobox(
            main_frame,
            textvariable=self.model_var,
            values=model_values,
            state="readonly" if len(model_values) > 0 else "disabled",
            width=15,
        )
        self.model_combo.grid(row=2, column=1, sticky="w", pady=(5, 0))

        # Predict button
        self.predict_button = ttk.Button(
            main_frame, text="Predict", command=self.on_predict
        )
        self.predict_button.grid(row=2, column=2, sticky="e", pady=(5, 0))

        # Result label
        self.result_label = ttk.Label(
            main_frame, textvariable=self.result_var, font=("Segoe UI", 11, "bold")
        )
        self.result_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # Make body expand with window
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        if self.baseline_model or self.bert_model:
            self.result_var.set("Models loaded. Enter news and click Predict.")
        else:
            self.result_var.set("No models could be loaded. Check outputs/ folder.")

    def on_predict(self):
        title = self.title_entry.get().strip()
        body = self.body_text.get("1.0", tk.END).strip()

        if not title and not body:
            messagebox.showwarning(
                "Input needed", "Please enter at least a title or a body."
            )
            return

        model_choice = self.model_var.get()
        if model_choice == "DistilBERT":
            model = self.bert_model
        elif model_choice == "Baseline":
            model = self.baseline_model
        else:
            messagebox.showerror(
                "No model", "No model is available. Check your outputs/ folder."
            )
            return

        if model is None:
            messagebox.showerror(
                "No model",
                f"The selected model '{model_choice}' is not loaded. Check your outputs/ folder.",
            )
            return

        try:
            pred_label, proba_fake = model.predict(title, body)
            label_str = "FAKE" if pred_label == 1 else "REAL"
            self.result_var.set(
                f"Prediction: {label_str}  (P(fake) = {proba_fake:.3f})"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            self.result_var.set("Prediction error.")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = FakeNewsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
