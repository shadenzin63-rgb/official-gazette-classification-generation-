# -*- coding: utf-8 -*-
"""
الخطوة 4: النموذج المتخصص في القطاعات الضعيفة
═══════════════════════════════════════════════════════
يُشغَّل بعد step2 بشكل مستقل

القطاعات المستهدفة (الأضعف من النتائج):
  - تشريعات وقرارات عليا  (F1=0.597, FP=74)
  - عمل وضمان اجتماعي    (F1=0.508)
  - أشغال وبنية تحتية    (F1=0.533)
  - تجارة وشركات         (F1=0.646)
  - مالية وضرائب         (F1=0.710)

الاستخدام:
  python step4_weak_specialist.py
  python step4_weak_specialist.py --resume --extra-epochs 10
"""

import matplotlib
matplotlib.use('Agg')

import os
import re
import gc
import json
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.amp import autocast, GradScaler
    def make_autocast(): return autocast(device_type='cuda')
    def make_scaler():   return GradScaler(device='cuda')
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    def make_autocast(): return autocast()
    def make_scaler():   return GradScaler()


# ═══════════════════════════════════════════════════════════════
# ⚙️ الإعدادات - عدّل المسارات
# ═══════════════════════════════════════════════════════════════

INPUT_FILE      = "/content/drive/MyDrive/Colab Notebooks/nlp/Augmented_Data_2_CLEAN.xlsx"
BASE_OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/nlp/temp2"
OUTPUT_DIR      = os.path.join(BASE_OUTPUT_DIR, "v5_weak_specialist")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

COLUMN_FILE_NAME = "اسم الملف"
COLUMN_SECTORS   = "القطاعات"
COLUMN_TEXT      = "text"

SECTOR_MERGE_MAP = {
    "طاقة وثروات":        "أشغال وبنية تحتية",
    "صناعة وتجارة":        "تجارة وشركات",
    "شؤون اجتماعية":       "عمل وضمان اجتماعي",
    "اتصالات وتكنولوجيا":  "إعلام ونشر",
}

# الـ 17 قطاع بنفس الترتيب الأصلي (ضروري للـ Ensemble)
ALL_SECTORS = [
    "أراضي وتنظيم",       "مالية وضرائب",          "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة",  "عقوبات وجرائم",          "أحوال شخصية",
    "عمل وضمان اجتماعي",  "تجارة وشركات",           "أشغال وبنية تحتية",
    "تعليم وبحث علمي",    "صحة وسلامة عامة",        "بيئة وزراعة",
    "أمن ودفاع",           "سياحة وآثار",            "إعلام ونشر",
    "نقل وسير",            "قضاء وتنفيذ"
]
SECTOR_TO_ID = {s: i for i, s in enumerate(ALL_SECTORS)}

# ── القطاعات المستهدفة مع إعدادات oversampling لكل منها ────────
TARGET_SECTORS = [
    "تشريعات وقرارات عليا",   # F1=0.597 - FP=74 (المشكلة الأكبر)
    "عمل وضمان اجتماعي",      # F1=0.508
    "أشغال وبنية تحتية",      # F1=0.533
    "تجارة وشركات",           # F1=0.646
    "مالية وضرائب",           # F1=0.710
]

# oversampling لكل قطاع حسب ندرته
OVERSAMPLING_CONFIG = {
    "أشغال وبنية تحتية":    5,   # نادر جداً (9 عينات test)
    "عمل وضمان اجتماعي":   3,   # نادر (28 عينة)
    "تجارة وشركات":         2,   # متوسط (48 عينة)
    "مالية وضرائب":         2,   # متوسط (55 عينة)
    "تشريعات وقرارات عليا": 1,   # كثير لكن مشكلته FP → downsampling بدل oversampling
}

# downsampling للتشريعات (هي المشكلة الأكبر)
LEGISLATION_MAX_RATIO = 0.20   # الحد الأقصى 20% من البيانات

OTHER_LABEL = "أخرى"

# النموذج: CaMeLBERT (أفضل نتيجة فردية 0.6955)
MODEL_NAME  = 'CAMeL-Lab/bert-base-arabic-camelbert-mix'
SHORT_NAME  = 'weak_specialist'
BERT_LR     = 8e-6    # LR أقل قليلاً للتخصص الدقيق
HEAD_LR     = 4e-5

MAX_LENGTH            = 384
BATCH_SIZE            = 4
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS            = 40
RANDOM_SEED           = 42
DEFAULT_THRESHOLD     = 0.50
HNM_START_EPOCH       = 3
HNM_WEIGHT            = 3.5   # أعلى قليلاً من step2 للتركيز على الأخطاء
HNM_TOP_PERCENT       = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 🔧 الدوال المساعدة
# ═══════════════════════════════════════════════════════════════

def parse_sectors(s):
    if pd.isna(s) or not isinstance(s, str):
        return []
    parts = [p.strip() for p in re.split(r'\s*[|¦,;/]\s*', s.strip()) if p.strip()]
    result = []
    for sec in parts:
        if sec in SECTOR_MERGE_MAP:
            t = SECTOR_MERGE_MAP[sec]
            if t in SECTOR_TO_ID:
                result.append(t)
        elif sec in SECTOR_TO_ID:
            result.append(sec)
    return list(set(result))


def truncate_long_text(text, max_words=600):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\b(عدد|العدد|الرقم|رقم|المادة|مادة|ماده|الى|إلى)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    chunk = max_words // 3
    mid   = len(words) // 2
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
    print(f"  مُزال: {initial - len(df)} من {initial}")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 🎯 إعداد البيانات المتخصصة
# ═══════════════════════════════════════════════════════════════

def downsample_legislation(df):
    """
    تقليل التشريعات لـ 20% من البيانات
    لأنها تسبب FP كثيرة (كانت 40%)
    """
    legislation_sec = "تشريعات وقرارات عليا"
    total     = len(df)
    max_count = int(total * LEGISLATION_MAX_RATIO)

    leg_only  = []   # وثائق تشريعات فقط (single-label)
    leg_multi = []   # وثائق تشريعات + قطاعات أخرى
    other     = []   # وثائق بدون تشريعات

    for idx, row in df.iterrows():
        secs = parse_sectors(str(row[COLUMN_SECTORS]))
        if legislation_sec in secs:
            if len(secs) == 1:
                leg_only.append(idx)
            else:
                leg_multi.append(idx)
        else:
            other.append(idx)

    current = len(leg_only) + len(leg_multi)
    if current <= max_count:
        print(f"  التشريعات ({current}) أقل من الحد ({max_count}) - لا تقليل")
        return df

    # نحتفظ بكل multi-label + عينات عشوائية من single-label
    keep_single = max(0, max_count - len(leg_multi))
    random.shuffle(leg_only)
    keep_indices = leg_multi + leg_only[:keep_single] + other
    df_new = df.loc[keep_indices].reset_index(drop=True)

    print(f"  Downsampling التشريعات: {current} → {len(leg_multi) + keep_single} "
          f"(نسبة {(len(leg_multi)+keep_single)/len(df_new)*100:.1f}%)")
    return df_new


def prepare_specialist_data(df):
    """
    تحضير البيانات للنموذج المتخصص:
    - القطاعات المستهدفة: label = 1
    - كل ما عداها: label "أخرى" = 1
    - oversampling للقطاعات النادرة
    """
    specialist_sectors = TARGET_SECTORS + [OTHER_LABEL]
    sec_to_idx = {s: i for i, s in enumerate(specialist_sectors)}
    n_labels   = len(specialist_sectors)

    texts, labels_list = [], []

    for _, row in df.iterrows():
        text       = str(row[COLUMN_TEXT])
        doc_secs   = parse_sectors(str(row[COLUMN_SECTORS]))
        label      = np.zeros(n_labels, dtype=np.float32)
        has_target = False

        for sec in doc_secs:
            if sec in sec_to_idx:
                label[sec_to_idx[sec]] = 1.0
                has_target = True

        if not has_target:
            label[sec_to_idx[OTHER_LABEL]] = 1.0

        texts.append(text)
        labels_list.append(label)

    texts  = np.array(texts)
    labels = np.array(labels_list, dtype=np.float32)

    # oversampling للقطاعات النادرة
    extra_texts, extra_labels = [], []
    for sec, copies in OVERSAMPLING_CONFIG.items():
        if copies <= 1 or sec not in sec_to_idx:
            continue
        i       = sec_to_idx[sec]
        mask    = labels[:, i] == 1.0
        n_pos   = mask.sum()
        if n_pos == 0:
            continue
        for _ in range(copies - 1):
            extra_texts.extend(texts[mask])
            extra_labels.extend(labels[mask])
        print(f"  oversampling {sec}: {n_pos} → {n_pos * copies}")

    if extra_texts:
        texts  = np.concatenate([texts,  np.array(extra_texts)])
        labels = np.concatenate([labels, np.array(extra_labels)])

    # خلط
    idx    = np.random.permutation(len(texts))
    texts  = texts[idx]
    labels = labels[idx]

    # طباعة التوزيع
    print(f"\n  توزيع بيانات المتخصص:")
    for i, sec in enumerate(specialist_sectors):
        n = int(labels[:, i].sum())
        print(f"    {sec:30s}: {n}")

    return texts, labels, specialist_sectors


# ═══════════════════════════════════════════════════════════════
# 📦 Dataset
# ═══════════════════════════════════════════════════════════════

class SpecialistDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts          = [str(t) for t in texts]
        self.labels         = labels
        self.tokenizer      = tokenizer
        self.sample_weights = np.ones(len(texts), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = truncate_long_text(self.texts[idx])
        enc  = self.tokenizer(text, max_length=MAX_LENGTH,
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
# 🧠 النموذج
# ═══════════════════════════════════════════════════════════════

class SpecialistClassifier(nn.Module):
    def __init__(self, model_name, n_labels, dropout=0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden          = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, n_labels)
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)


# ═══════════════════════════════════════════════════════════════
# ⚖️  Focal Loss + Class Weights
# ═══════════════════════════════════════════════════════════════

def calculate_class_weights(labels, max_w=15.0, beta=0.9999):
    pos = labels.sum(axis=0).astype(np.float64)
    eff = 1.0 - np.power(beta, pos)
    w   = (1.0 - beta) / (eff + 1e-8)
    w   = w / w.mean()
    return torch.tensor(np.clip(w, 0.3, max_w), dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets, sample_weights=None):
        t_s  = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, t_s, pos_weight=self.pos_weight, reduction='none')
        p    = torch.sigmoid(logits)
        p_t  = p * targets + (1 - p) * (1 - targets)
        loss = (1 - p_t).pow(self.gamma) * bce
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════
# 🎯 Hard Negative Mining
# ═══════════════════════════════════════════════════════════════

def update_hnm_weights(model, dataset, device):
    model.eval()
    losses = np.zeros(len(dataset), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in tqdm(loader, desc="HNM", leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            idxs = batch['idx'].numpy()
            with make_autocast():
                logits = model(ids, mask)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, lbls, reduction='none')
            for i, idx in enumerate(idxs):
                losses[idx] = bce[i].mean().item()

    threshold              = np.sort(losses)[-int(len(losses) * HNM_TOP_PERCENT)]
    weights                = np.ones(len(losses), dtype=np.float32)
    hard_mask              = losses >= threshold
    weights[hard_mask]     = HNM_WEIGHT
    dataset.sample_weights = weights
    print(f"  HNM: {hard_mask.sum()} عينة صعبة (x{HNM_WEIGHT})")


# ═══════════════════════════════════════════════════════════════
# 📊 Threshold Tuning
# ═══════════════════════════════════════════════════════════════

def find_thresholds(probs, labels, specialist_sectors):
    n_labels   = len(specialist_sectors)
    thresholds = np.full(n_labels, DEFAULT_THRESHOLD, dtype=np.float32)
    other_idx  = specialist_sectors.index(OTHER_LABEL)
    total      = len(labels)

    for i, sec in enumerate(specialist_sectors):
        if i == other_idx:
            thresholds[i] = 0.50
            continue

        yt, prob = labels[:, i], probs[:, i]
        n_pos    = int(yt.sum())
        if n_pos == 0:
            continue

        base_f1      = f1_score(yt, (prob > 0.50).astype(int), zero_division=0)
        best_f1, best_th = base_f1, 0.50
        prevalence   = n_pos / total

        # للتشريعات: ابدأ البحث من 0.40 للحد من FP
        start_th = 0.40 if sec == "تشريعات وقرارات عليا" else 0.15

        for th in np.arange(start_th, 0.85, 0.02):
            pred = (prob > th).astype(int)
            f    = f1_score(yt, pred, zero_division=0)
            # penalty أقوى على over-prediction للتشريعات
            if prevalence > 0.15 and pred.mean() > prevalence * 1.5:
                f *= 0.90
            if f > best_f1 + 0.005:
                best_f1, best_th = f, th

        # قطاعات نادرة: لا تتجاوز 0.45
        if n_pos < 15:
            best_th = min(best_th, 0.45)

        thresholds[i] = float(np.clip(best_th, 0.15, 0.85))
        print(f"    threshold {sec:30s}: {thresholds[i]:.2f}  (F1={best_f1:.3f})")

    return thresholds


# ═══════════════════════════════════════════════════════════════
# 📊 Evaluate
# ═══════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device, thresholds):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            with make_autocast():
                logits = model(ids, mask)
                loss   = criterion(logits, lbls)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > thresholds[np.newaxis, :]).astype(float)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(lbls.cpu().numpy())

    probs  = np.vstack(all_probs)
    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    f1     = f1_score(labels, preds, average='macro', zero_division=0)
    return f1, probs, labels, total_loss / max(1, len(loader))


# ═══════════════════════════════════════════════════════════════
# 🔄 تحويل مخرجات المتخصص إلى شكل الـ 17 قطاع
# ═══════════════════════════════════════════════════════════════

def expand_to_full_sectors(specialist_probs, specialist_sectors):
    """
    يحول probs المتخصص (n × 6) إلى شكل الـ 17 قطاع (n × 17)
    القطاعات غير الموجودة في المتخصص تبقى 0.0
    step3 سيستخدم هذا الملف للـ Ensemble
    """
    n_samples  = specialist_probs.shape[0]
    full_probs = np.zeros((n_samples, len(ALL_SECTORS)), dtype=np.float32)

    for i, sec in enumerate(specialist_sectors):
        if sec == OTHER_LABEL:
            continue
        if sec in SECTOR_TO_ID:
            full_probs[:, SECTOR_TO_ID[sec]] = specialist_probs[:, i]

    return full_probs


# ═══════════════════════════════════════════════════════════════
# 🏋️  Training Loop
# ═══════════════════════════════════════════════════════════════

def train(resume: bool = False, extra_epochs: int = 10):
    print("=" * 70)
    print(f"النموذج الرابع المتخصص: {MODEL_NAME}")
    print(f"القطاعات: {TARGET_SECTORS}")
    print(f"الجهاز: {DEVICE}")
    print("=" * 70)

    # ── تحميل البيانات ───────────────────────────────────────
    print(f"\nتحميل البيانات...")
    try:
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    except FileNotFoundError:
        fallback = "/home/albara/Desktop/project NLP/data/Final_Merged_to_classification.xlsx"
        print(f"  Augmented غير موجود، استخدام: {fallback}")
        df = pd.read_excel(fallback, engine='openpyxl')

    df = remove_duplicates(df)
    print(f"  {len(df)} وثيقة")

    # ── Downsampling للتشريعات ───────────────────────────────
    print(f"\nDownsampling التشريعات...")
    df = downsample_legislation(df)

    # ── تحضير بيانات المتخصص ────────────────────────────────
    print(f"\nتحضير بيانات المتخصص...")
    texts, labels, specialist_sectors = prepare_specialist_data(df)
    other_idx = specialist_sectors.index(OTHER_LABEL)
    n_labels  = len(specialist_sectors)

    print(f"\n  عدد القطاعات (مع أخرى): {n_labels}")
    print(f"  إجمالي العينات بعد oversampling: {len(texts)}")

    # ── تقسيم ───────────────────────────────────────────────
    y_main = labels.argmax(axis=1)
    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            texts, labels, test_size=0.20,
            random_state=RANDOM_SEED, stratify=y_main)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50,
            random_state=RANDOM_SEED, stratify=y_tmp.argmax(axis=1))
    except ValueError:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            texts, labels, test_size=0.20, random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED)

    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Tokenizer + DataLoaders ──────────────────────────────
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds     = SpecialistDataset(X_tr,   y_tr,   tokenizer)
    val_ds       = SpecialistDataset(X_val,  y_val,  tokenizer)
    test_ds      = SpecialistDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── النموذج + Loss + Optimizer ───────────────────────────
    model     = SpecialistClassifier(MODEL_NAME, n_labels).to(DEVICE)
    pos_w     = calculate_class_weights(y_tr).to(DEVICE)
    criterion = FocalLoss(pos_w, gamma=2.0, smoothing=0.05)

    no_decay  = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer = AdamW([
        {'params': [p for n, p in model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'lr': BERT_LR, 'weight_decay': 0.01},
        {'params': [p for n, p in model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'lr': BERT_LR, 'weight_decay': 0.0},
        {'params': [p for n, p in model.classifier.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'lr': HEAD_LR, 'weight_decay': 0.01},
        {'params': [p for n, p in model.classifier.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'lr': HEAD_LR, 'weight_decay': 0.0},
    ], eps=1e-8)

    num_epochs_local = NUM_EPOCHS
    steps_per_epoch  = max(1, len(train_loader) // GRADIENT_ACCUMULATION)
    total_steps      = steps_per_epoch * num_epochs_local
    scheduler        = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    scaler           = make_scaler()
    best_f1          = 0.0
    thresholds       = np.full(n_labels, DEFAULT_THRESHOLD, dtype=np.float32)
    patience_counter = 0
    patience         = 7
    start_epoch      = 0

    # ── استئناف من checkpoint ────────────────────────────────
    if resume and os.path.exists(BEST_MODEL_PATH):
        torch.cuda.empty_cache(); gc.collect()
        ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(DEVICE)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch      = ckpt['epoch'] + 1
        best_f1          = ckpt['f1_macro']
        thresholds       = np.array(ckpt['thresholds'], dtype=np.float32)
        num_epochs_local = start_epoch + extra_epochs
        print(f"  استئناف من epoch {start_epoch}, F1={best_f1:.4f}")

    # ── Training Loop ────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs_local):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs_local} [{SHORT_NAME}]")
        print('='*70)

        if epoch >= HNM_START_EPOCH:
            update_hnm_weights(model, train_ds, DEVICE)

        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            sw   = batch['weight'].to(DEVICE)

            with make_autocast():
                logits = model(ids, mask)
                loss   = criterion(logits, lbls, sw) / GRADIENT_ACCUMULATION

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (step+1) % GRADIENT_ACCUMULATION == 0 or (step+1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n  Train Loss: {avg_loss:.4f}")

        current_f1, val_probs, val_labels_ep, _ = evaluate(
            model, val_loader, criterion, DEVICE, thresholds)

        print(f"  F1-Macro: {current_f1:.4f}  |  Best: {best_f1:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if current_f1 > best_f1:
            best_f1    = current_f1
            print(f"\n  Threshold Tuning...")
            thresholds = find_thresholds(val_probs, val_labels_ep, specialist_sectors)
            patience_counter = 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_macro':            best_f1,
                'thresholds':          thresholds.tolist(),
                'specialist_sectors':  specialist_sectors,
                'model_name':          MODEL_NAME,
            }, BEST_MODEL_PATH)
            print(f"  محفوظ (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  لا تحسّن ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early Stopping! Best F1: {best_f1:.4f}")
                break

    # ── تقييم على Test Set ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"تقييم Test Set [{SHORT_NAME}]")
    print('='*70)

    torch.cuda.empty_cache(); gc.collect()
    ckpt = torch.load(BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model      = model.to(DEVICE)
    thresholds = np.array(ckpt['thresholds'], dtype=np.float32)

    test_f1, test_probs, test_labels_ts, _ = evaluate(
        model, test_loader, criterion, DEVICE, thresholds)

    # تفصيل لكل قطاع مستهدف
    print(f"\n  Test F1-Macro (القطاعات المستهدفة): {test_f1:.4f}\n")
    print(f"  {'القطاع':30s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s}")
    print("  " + "─" * 52)
    for i, sec in enumerate(specialist_sectors):
        if sec == OTHER_LABEL:
            continue
        yt  = test_labels_ts[:, i]
        yp  = (test_probs[:, i] > thresholds[i]).astype(int)
        f   = f1_score(yt, yp, zero_division=0)
        p   = precision_score(yt, yp, zero_division=0)
        r   = recall_score(yt, yp, zero_division=0)
        print(f"  {sec:30s} {f:.4f} {p:.4f} {r:.4f}")

    # ── تحويل وحفظ للـ Ensemble ──────────────────────────────
    full_test_probs = expand_to_full_sectors(test_probs, specialist_sectors)
    full_val_probs  = evaluate(model, val_loader, criterion, DEVICE, thresholds)[1]
    full_val_probs  = expand_to_full_sectors(full_val_probs, specialist_sectors)

    np.save(os.path.join(OUTPUT_DIR, 'test_probs.npy'), full_test_probs)
    np.save(os.path.join(OUTPUT_DIR, 'val_probs.npy'),  full_val_probs)

    # حفظ النتائج
    results = {
        'model':               MODEL_NAME,
        'short_name':          SHORT_NAME,
        'target_sectors':      TARGET_SECTORS,
        'specialist_sectors':  specialist_sectors,
        'test_f1_specialist':  round(test_f1, 4),
        'thresholds':          thresholds.tolist(),
        'completed':           datetime.now().isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, 'final_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'tokenizer'))

    print(f"\n  مكتمل! F1: {test_f1:.4f}")
    print(f"  test_probs.npy محفوظ (shape: {full_test_probs.shape})")
    print(f"  المخرجات في: {OUTPUT_DIR}")
    print(f"\n  الخطوة التالية:")
    print(f"  python step3_ensemble_v2.py --method all")


# ═══════════════════════════════════════════════════════════════
# 🎯 Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--extra-epochs', type=int, default=10)
    args = parser.parse_args()

    train(resume=args.resume, extra_epochs=args.extra_epochs)
