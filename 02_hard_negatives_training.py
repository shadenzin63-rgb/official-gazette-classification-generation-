# -*- coding: utf-8 -*-
"""
الخطوة 2: التدريب المحسّن مع Hard Negative Mining
═══════════════════════════════════════════════════════
يُشغَّل بعد step1_data_augmentation.py

الميزات الجديدة:
  1. Hard Negative Mining: تركيز إضافي على العينات الصعبة
  2. دعم نماذج متعددة: AraBERT v2, MarBERT, CAMeLBERT
  3. Focal Loss محسّن مع gamma تكيّفي
  4. Cosine LR + Warmup
  5. حفظ probabilities للـ Ensemble

الاستخدام:
  python step2_train_with_hard_negatives.py --model arabert
  python step2_train_with_hard_negatives.py --model marbert
  python step2_train_with_hard_negatives.py --model camelbert
"""

import matplotlib
matplotlib.use('Agg')

import os
import re
import json
import random
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.amp import autocast, GradScaler
    def make_autocast():
        return autocast(device_type='cuda')
    def make_scaler():
        return GradScaler(device='cuda')
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    def make_autocast():
        return autocast()
    def make_scaler():
        return GradScaler()


# ═══════════════════════════════════════════════════════════════
# ⚙️ الإعدادات
# ═══════════════════════════════════════════════════════════════

# النماذج المتاحة
MODEL_CONFIGS = {
    'arabert': {
        'name':       'aubmindlab/bert-base-arabertv2',
        'short_name': 'arabert_v2',
        'bert_lr':    1e-5,
        'head_lr':    5e-5,
    },
    'marbert': {
        'name':       'UBC-NLP/MARBERT',
        'short_name': 'marbert',
        'bert_lr':    1e-5,
        'head_lr':    5e-5,
    },
    'camelbert': {
        'name':       'CAMeL-Lab/bert-base-arabic-camelbert-mix',
        'short_name': 'camelbert_mix',
        'bert_lr':    1e-5,
        'head_lr':    5e-5,
    },
}

# ← عدّل هذه المسارات
INPUT_FILE = "/content/drive/MyDrive/Colab Notebooks/nlp/Augmented_Data_2_CLEAN.xlsx"
BASE_OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/nlp/temp2"

COLUMN_FILE_NAME = "اسم الملف"
COLUMN_SECTORS   = "القطاعات"
COLUMN_TEXT      = "text"

SECTOR_MERGE_MAP = {
    "طاقة وثروات":        "أشغال وبنية تحتية",
    "صناعة وتجارة":        "تجارة وشركات",
    "شؤون اجتماعية":       "عمل وضمان اجتماعي",
    "اتصالات وتكنولوجيا":  "إعلام ونشر",
}

ALL_SECTORS = [
    "أراضي وتنظيم",       "مالية وضرائب",          "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة",  "عقوبات وجرائم",          "أحوال شخصية",
    "عمل وضمان اجتماعي",  "تجارة وشركات",           "أشغال وبنية تحتية",
    "تعليم وبحث علمي",    "صحة وسلامة عامة",        "بيئة وزراعة",
    "أمن ودفاع",           "سياحة وآثار",            "إعلام ونشر",
    "نقل وسير",            "قضاء وتنفيذ"
]
SECTOR_TO_ID = {s: i for i, s in enumerate(ALL_SECTORS)}
ID_TO_SECTOR = {i: s for s, i in SECTOR_TO_ID.items()}

MAX_LENGTH            = 384      # كان 512 ← أسرع 25%
BATCH_SIZE            = 2        # ابق كما هو
GRADIENT_ACCUMULATION = 8        # كان 16 ← effective batch = 16 يكفي
NUM_EPOCHS            = 50       # كان 40 ← Early Stopping سيوقف قبل
RANDOM_SEED           = 42
DEFAULT_THRESHOLD     = 0.50
TOP_K                 = 2

# Hard Negative Mining
HNM_START_EPOCH  = 3     # ابدأ HNM بعد 3 epochs (دع النموذج يتعلم أولاً)
HNM_WEIGHT       = 3.0   # وزن إضافي للعينات الصعبة
HNM_TOP_PERCENT  = 0.25  # نسبة أصعب العينات

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════
# 🔧 الدوال المساعدة
# ═══════════════════════════════════════════════════════════════

def parse_sectors(s):
    if pd.isna(s) or not isinstance(s, str):
        return []
    sectors = [p.strip() for p in re.split(r'\s*[|¦,;/]\s*', s.strip()) if p.strip()]
    merged = []
    for sec in sectors:
        if sec in SECTOR_MERGE_MAP:
            t = SECTOR_MERGE_MAP[sec]
            if t in SECTOR_TO_ID:
                merged.append(t)
        elif sec in SECTOR_TO_ID:
            merged.append(sec)
    return list(set(merged))


def create_multi_label_encoding(df):
    labels = np.zeros((len(df), len(ALL_SECTORS)), dtype=np.float32)
    for i in range(len(df)):
        for sec in parse_sectors(str(df[COLUMN_SECTORS].iloc[i])):
            if sec in SECTOR_TO_ID:
                labels[i, SECTOR_TO_ID[sec]] = 1.0
    return labels


def truncate_long_text(text, max_words=2000):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # إزالة الكلمات الشكلية القانونية
    text = re.sub(r'\b(عدد|العدد|الرقم|رقم|المادة|مادة|ماده|الى|إلى)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.strip().split()
    if len(words) <= max_words:
        return text
    chunk = max_words // 3
    mid = len(words) // 2
    return ' '.join(words[:chunk] + ['[...]'] +
                    words[mid-chunk//2:mid+chunk//2] + ['[...]'] +
                    words[-chunk:])


def remove_duplicates(df):
    initial = len(df)
    if COLUMN_FILE_NAME in df.columns:
        df = df.drop_duplicates(subset=[COLUMN_FILE_NAME], keep='first')
    df = df.copy()
    df['_sig'] = df[COLUMN_TEXT].apply(
        lambda x: ' '.join(str(x).strip().split()[:30]).lower() if pd.notna(x) else "")
    df = df.drop_duplicates(subset=['_sig'], keep='first').drop(columns=['_sig'])
    print(f"  🗑️  مُزال: {initial - len(df)} من {initial}")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 📦 Dataset
# ═══════════════════════════════════════════════════════════════

class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = [str(t) if pd.notna(t) else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # sample weights (يُحدَّث بـ Hard Negative Mining)
        self.sample_weights = np.ones(len(texts), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = truncate_long_text(self.texts[idx], max_words=600)
        enc = self.tokenizer(text, max_length=self.max_length,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float32),
            'weight':         torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            'idx':            torch.tensor(idx, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════
# 🧠 النموذج (يدعم نماذج مختلفة)
# ═══════════════════════════════════════════════════════════════

class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, n_labels, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.classifier(pooled)


# ═══════════════════════════════════════════════════════════════
# ⚖️  Class Weights + Focal Loss
# ═══════════════════════════════════════════════════════════════

def calculate_class_weights(labels, max_w=12.0, beta=0.9999):
    pos = labels.sum(axis=0).astype(np.float64)
    eff = 1.0 - np.power(beta, pos)
    w = (1.0 - beta) / (eff + 1e-8)
    w = w / w.mean()
    w = np.clip(w, 0.3, max_w)
    return torch.tensor(w, dtype=torch.float32)


class FocalLossWithHNM(nn.Module):
    """
    Focal Loss مع دعم Hard Negative Mining:
    - gamma: يركز على العينات الصعبة
    - sample_weight: وزن إضافي لكل عينة (من HNM)
    """
    def __init__(self, pos_weight, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets, sample_weights=None):
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_s, pos_weight=self.pos_weight, reduction='none')

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t).pow(self.gamma) * bce

        if sample_weights is not None:
            # [batch] → [batch, 1] للبث
            focal = focal * sample_weights.unsqueeze(1)

        return focal.mean()


# ═══════════════════════════════════════════════════════════════
# 🎯 Hard Negative Mining
# ═══════════════════════════════════════════════════════════════

def update_hard_negative_weights(model, train_dataset, criterion, device,
                                  top_percent=HNM_TOP_PERCENT,
                                  hn_weight=HNM_WEIGHT):
    """
    حساب الخسارة لكل عينة تدريب وتعيين أوزان أعلى للعينات الصعبة.
    يُستدعى بعد كل epoch بدءاً من HNM_START_EPOCH.
    """
    model.eval()
    losses = np.zeros(len(train_dataset), dtype=np.float32)

    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(loader, desc="HNM scoring", leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            idxs = batch['idx'].numpy()

            with make_autocast():
                logits = model(ids, mask)
            # حساب الخسارة لكل عينة
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, lbls, reduction='none')
            per_sample_loss = bce.mean(dim=1).cpu().numpy()

            for i, idx in enumerate(idxs):
                losses[idx] = per_sample_loss[i]

    # تحديد أصعب العينات (أعلى خسارة)
    n_hard = int(len(losses) * top_percent)
    threshold_loss = np.sort(losses)[-n_hard] if n_hard > 0 else losses.max()

    # تعيين الأوزان
    new_weights = np.ones(len(losses), dtype=np.float32)
    hard_mask = losses >= threshold_loss
    new_weights[hard_mask] = hn_weight

    train_dataset.sample_weights = new_weights

    n_hard_actual = hard_mask.sum()
    avg_hard_loss = losses[hard_mask].mean() if n_hard_actual > 0 else 0
    avg_easy_loss = losses[~hard_mask].mean()

    print(f"  🎯 HNM: {n_hard_actual} عينات صعبة (weight={hn_weight}x)")
    print(f"     avg_hard_loss={avg_hard_loss:.4f}, avg_easy_loss={avg_easy_loss:.4f}")

    return new_weights


# ═══════════════════════════════════════════════════════════════
# 📊 Metrics + Thresholds
# ═══════════════════════════════════════════════════════════════

def calculate_metrics(y_true, y_pred, y_probs=None):
    m = {
        'accuracy_exact_match': float(accuracy_score(y_true, y_pred)),
        'f1_micro':    float(f1_score(y_true, y_pred, average='micro',    zero_division=0)),
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro':    float(recall_score(y_true,    y_pred, average='macro', zero_division=0)),
    }
    m['per_sector'] = {}
    for i, sec in enumerate(ALL_SECTORS):
        yt, yp = y_true[:, i], y_pred[:, i]
        p = float(precision_score(yt, yp, zero_division=0))
        r = float(recall_score(yt, yp, zero_division=0))
        f = float(f1_score(yt, yp, zero_division=0))
        m['per_sector'][sec] = {
            'precision': round(p, 4), 'recall': round(r, 4),
            'f1': round(f, 4), 'support': int(yt.sum())
        }
    if y_probs is not None:
        m['error_analysis'] = {}
        for i, sec in enumerate(ALL_SECTORS):
            yt, yp, prob = y_true[:, i], y_pred[:, i], y_probs[:, i]
            fp = int(np.sum((yt == 0) & (yp == 1)))
            fn = int(np.sum((yt == 1) & (yp == 0)))
            m['error_analysis'][sec] = {
                'false_positives': fp, 'false_negatives': fn,
                'avg_fp_confidence': round(float(np.mean(prob[(yt==0)&(yp==1)])) if fp>0 else 0, 3),
                'avg_fn_confidence': round(float(np.mean(prob[(yt==1)&(yp==0)])) if fn>0 else 0, 3),
            }
    return m


def find_optimal_thresholds(probs, labels):
    thresholds = np.full(len(ALL_SECTORS), DEFAULT_THRESHOLD, dtype=np.float32)
    total = len(labels)
    for i in range(len(ALL_SECTORS)):
        yt, prob = labels[:, i], probs[:, i]
        n_pos = int(yt.sum())
        if n_pos == 0:
            continue
        base_f1 = f1_score(yt, (prob > DEFAULT_THRESHOLD).astype(int), zero_division=0)
        best_f1, best_th = base_f1, DEFAULT_THRESHOLD
        prevalence = n_pos / total
        for th in np.arange(0.15, 0.85, 0.02):
            pred = (prob > th).astype(int)
            f1 = f1_score(yt, pred, zero_division=0)
            if prevalence > 0.15:
                pred_rate = pred.mean()
                if pred_rate > prevalence * 1.8:
                    f1 *= 0.95
            if f1 > best_f1 + 0.005:
                best_f1, best_th = f1, th
        if n_pos < 15:
            best_th = min(best_th, 0.45)
        thresholds[i] = float(np.clip(best_th, 0.15, 0.85))
    return thresholds


def evaluate(model, dataloader, criterion, device, thresholds=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    if thresholds is None:
        thresholds = np.full(len(ALL_SECTORS), DEFAULT_THRESHOLD, dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            with make_autocast():
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            preds = (probs > thresholds[np.newaxis, :]).astype(np.float32)
            all_preds.append(preds)
            all_labels.append(lbls.cpu().numpy())
            all_probs.append(probs)
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['avg_loss'] = total_loss / max(1, len(dataloader))
    return metrics, all_probs, all_labels


# ═══════════════════════════════════════════════════════════════
# 🏋️  Training Loop مع Hard Negative Mining
# ═══════════════════════════════════════════════════════════════

def train(model_key: str, resume: bool = False, extra_epochs: int = 10):
    config = MODEL_CONFIGS[model_key]
    model_name = config['name']
    short_name = config['short_name']

    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"v5_{short_name}")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"🚀 التدريب: {model_name}")
    print(f"📤 المخرجات: {OUTPUT_DIR}")
    print(f"🖥️  الجهاز: {DEVICE}")
    print("=" * 70)

    # ── تحميل البيانات ───────────────────────────────────────
    print(f"\n📂 تحميل البيانات...")
    try:
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    except FileNotFoundError:
        # fallback إلى الملف الأصلي
        fallback = "/home/albara/Desktop/project NLP/data/Final_Merged_to_classification.xlsx"
        print(f"  ⚠️  الملف الموسّع غير موجود، استخدام: {fallback}")
        df = pd.read_excel(fallback, engine='openpyxl')

    df = remove_duplicates(df)
    labels = create_multi_label_encoding(df)
    print(f"  ✅ {len(df)} وثيقة, {len(ALL_SECTORS)} قطاع")

    # ── تقسيم البيانات ───────────────────────────────────────
    y_main = labels.argmax(axis=1)
    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            df[COLUMN_TEXT].values, labels,
            test_size=0.20, random_state=RANDOM_SEED, stratify=y_main)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED,
            stratify=y_tmp.argmax(axis=1))
    except ValueError:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            df[COLUMN_TEXT].values, labels,
            test_size=0.20, random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED)

    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Tokenizer + DataLoaders ──────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = LegalDataset(X_tr, y_tr, tokenizer, MAX_LENGTH)
    val_dataset   = LegalDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset  = LegalDataset(X_test, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── النموذج + Loss + Optimizer ───────────────────────────
    model = MultiLabelClassifier(model_name, n_labels=len(ALL_SECTORS)).to(DEVICE)
    pos_w = calculate_class_weights(y_tr).to(DEVICE)
    criterion = FocalLossWithHNM(pos_w, gamma=2.0, smoothing=0.05)

    # AdamW مع weight decay صحيح
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    bert_params = list(model.bert.named_parameters())
    head_params = list(model.classifier.named_parameters())
    optimizer = AdamW([
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
         'lr': config['bert_lr'], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
         'lr': config['bert_lr'], 'weight_decay': 0.0},
        {'params': [p for n, p in head_params if not any(nd in n for nd in no_decay)],
         'lr': config['head_lr'], 'weight_decay': 0.01},
        {'params': [p for n, p in head_params if any(nd in n for nd in no_decay)],
         'lr': config['head_lr'], 'weight_decay': 0.0},
    ], eps=1e-8)

    num_epochs_local = NUM_EPOCHS
    steps_per_epoch = max(1, len(train_loader) // GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * num_epochs_local
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    scaler = make_scaler()
    best_f1 = 0.0
    thresholds = np.full(len(ALL_SECTORS), DEFAULT_THRESHOLD, dtype=np.float32)
    patience_counter = 0
    patience = 7
    start_epoch = 0

    # ── استئناف من checkpoint ────────────────────────────────
    if resume and os.path.exists(BEST_MODEL_PATH):
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(DEVICE)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt['metrics']['f1_macro']
        thresholds = np.array(ckpt['thresholds'], dtype=np.float32)
        num_epochs_local = start_epoch + extra_epochs
        print(f"  🔄 استئناف من epoch {start_epoch}, F1={best_f1:.4f}")
        print(f"  🎯 التدريب حتى epoch {NUM_EPOCHS}")
    else:
        print(f"  🆕 بدء تدريب جديد حتى epoch {NUM_EPOCHS}")

    # ── Training Loop ────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs_local):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs_local} [{short_name}]")
        print('='*70)

        # ── Hard Negative Mining (بعد epoch 3) ──
        if epoch >= HNM_START_EPOCH:
            update_hard_negative_weights(
                model, train_dataset, criterion, DEVICE,
                top_percent=HNM_TOP_PERCENT, hn_weight=HNM_WEIGHT)

        # ── Training ──
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            lbls   = batch['labels'].to(DEVICE)
            sw     = batch['weight'].to(DEVICE)

            with make_autocast():
                logits = model(ids, mask)
                loss = criterion(logits, lbls, sample_weights=sw) / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n  📉 Train Loss: {avg_loss:.4f}")

        # ── Validation ──
        val_metrics, val_probs, val_labels = evaluate(
            model, val_loader, criterion, DEVICE, thresholds)
        current_f1 = val_metrics['f1_macro']

        print(f"  F1-Macro : {current_f1:.4f}  ⭐")
        print(f"  Precision: {val_metrics['precision_macro']:.4f}")
        print(f"  Recall   : {val_metrics['recall_macro']:.4f}")
        print(f"  Best     : {best_f1:.4f}")
        print(f"  LR       : {optimizer.param_groups[0]['lr']:.2e}")

        # ── حفظ أفضل نموذج ──
        if current_f1 > best_f1:
            improvement = current_f1 - best_f1
            best_f1 = current_f1
            thresholds = find_optimal_thresholds(val_probs, val_labels)
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'thresholds': thresholds.tolist(),
                'model_name': model_name,
            }, BEST_MODEL_PATH)
            print(f"  ✅ تحسّن +{improvement:.4f} → محفوظ")
        else:
            patience_counter += 1
            print(f"  🚫 لا تحسّن ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  🛑 Early Stopping! Best F1-Macro: {best_f1:.4f}")
                break

    # ── تقييم على Test Set ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🧪 تقييم على Test Set [{short_name}]")
    print('='*70)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    thresholds = np.array(ckpt['thresholds'], dtype=np.float32)

    test_metrics, test_probs, test_labels = evaluate(
        model, test_loader, criterion, DEVICE, thresholds)

    print(f"\n  Test F1-Macro    : {test_metrics['f1_macro']:.4f}  ⭐")
    print(f"  Test Precision   : {test_metrics['precision_macro']:.4f}")
    print(f"  Test Recall      : {test_metrics['recall_macro']:.4f}")
    print(f"  Test Accuracy    : {test_metrics['accuracy_exact_match']:.4f}")

    # حفظ النتائج + probabilities للـ Ensemble
    results = {
        'model': model_name,
        'test_metrics': test_metrics,
        'thresholds': thresholds.tolist(),
        'completed': datetime.now().isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # حفظ probabilities للـ Ensemble
    np.save(os.path.join(OUTPUT_DIR, 'test_probs.npy'), test_probs)
    np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_labels)
    np.save(os.path.join(OUTPUT_DIR, 'val_probs.npy'),
            evaluate(model, val_loader, criterion, DEVICE, thresholds)[1])
    np.save(os.path.join(OUTPUT_DIR, 'val_labels.npy'), y_val)

    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))

    print(f"\n✅ التدريب اكتمل! [{short_name}] F1-Macro: {test_metrics['f1_macro']:.4f}")
    return test_metrics


# ═══════════════════════════════════════════════════════════════
# 🎯 Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='arabert',
                        choices=['arabert', 'marbert', 'camelbert'],
                        help='اختر النموذج')
    parser.add_argument('--resume', action='store_true', help='استئناف من checkpoint')
    parser.add_argument('--extra-epochs', type=int, default=10, help='عدد epochs إضافية')
    args = parser.parse_args()

    print(f"\n🚀 النموذج المختار: {args.model}")
    train(args.model, resume=args.resume, extra_epochs=args.extra_epochs)