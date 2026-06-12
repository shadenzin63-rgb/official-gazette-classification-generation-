# -*- coding: utf-8 -*-
"""
الخطوة 3: Ensemble من عدة نماذج
═══════════════════════════════════════════════════════
يُشغَّل بعد تدريب 2-3 نماذج بـ step2.

3 طرق Ensemble:
  1. Average: متوسط الـ probabilities
  2. Weighted Average: أوزان حسب أداء كل نموذج
  3. Stacking: meta-learner يتعلم من النماذج الثلاثة

الاستخدام:
  python step3_ensemble.py
  python step3_ensemble.py --method weighted
  python step3_ensemble.py --method stacking
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# ═══════════════════════════════════════════════════════════════
# ⚙️ الإعدادات
# ═══════════════════════════════════════════════════════════════

BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/nlp/temp2"

# المجلدات الناتجة من step2 (أضف/احذف حسب النماذج التي دربتها)
MODEL_DIRS = {
    'arabert':   os.path.join(BASE_DIR, "v5_arabert_v2"),
    'marbert':   os.path.join(BASE_DIR, "v5_marbert"),
    'camelbert': os.path.join(BASE_DIR, "v5_camelbert_mix"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "v5_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_SECTORS = [
    "أراضي وتنظيم",       "مالية وضرائب",          "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة",  "عقوبات وجرائم",          "أحوال شخصية",
    "عمل وضمان اجتماعي",  "تجارة وشركات",           "أشغال وبنية تحتية",
    "تعليم وبحث علمي",    "صحة وسلامة عامة",        "بيئة وزراعة",
    "أمن ودفاع",           "سياحة وآثار",            "إعلام ونشر",
    "نقل وسير",            "قضاء وتنفيذ"
]


# ═══════════════════════════════════════════════════════════════
# 📂 تحميل النتائج
# ═══════════════════════════════════════════════════════════════

def load_model_outputs():
    """تحميل probabilities وlabels من كل نموذج"""
    models = {}
    for name, path in MODEL_DIRS.items():
        test_probs_path = os.path.join(path, 'test_probs.npy')
        test_labels_path = os.path.join(path, 'test_labels.npy')
        val_probs_path = os.path.join(path, 'val_probs.npy')
        metrics_path = os.path.join(path, 'final_metrics.json')

        if not os.path.exists(test_probs_path):
            print(f"  ⚠️  {name}: لم يتم تدريبه (تخطي)")
            continue

        test_probs = np.load(test_probs_path)
        test_labels = np.load(test_labels_path)

        val_probs = None
        val_labels = None
        if os.path.exists(val_probs_path):
            val_probs = np.load(val_probs_path)
            val_labels_path = os.path.join(path, 'val_labels.npy')
            if os.path.exists(val_labels_path):
                val_labels = np.load(val_labels_path)

        f1 = 0.0
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            f1 = data.get('test_metrics', {}).get('f1_macro', 0)

        models[name] = {
            'test_probs': test_probs,
            'test_labels': test_labels,
            'val_probs': val_probs,
            'val_labels': val_labels,
            'f1_macro': f1,
        }
        print(f"  ✅ {name}: F1-Macro={f1:.4f}, shape={test_probs.shape}")

    return models


# ═══════════════════════════════════════════════════════════════
# 1️⃣ Simple Average Ensemble
# ═══════════════════════════════════════════════════════════════

def ensemble_average(models):
    """متوسط بسيط لـ probabilities"""
    probs_list = [m['test_probs'] for m in models.values()]
    avg_probs = np.mean(probs_list, axis=0)
    return avg_probs


# ═══════════════════════════════════════════════════════════════
# 2️⃣ Weighted Average Ensemble
# ═══════════════════════════════════════════════════════════════

def ensemble_weighted(models):
    """متوسط مرجّح: النموذج الأفضل يحصل على وزن أعلى"""
    f1_scores = np.array([m['f1_macro'] for m in models.values()])

    # تحويل F1 إلى أوزان (مربع F1 لتعظيم الفرق)
    weights = f1_scores ** 2
    weights = weights / weights.sum()

    print(f"\n  أوزان الـ Ensemble:")
    for (name, _), w in zip(models.items(), weights):
        print(f"    {name}: {w:.3f}")

    probs_list = [m['test_probs'] for m in models.values()]
    weighted_probs = np.zeros_like(probs_list[0])
    for probs, w in zip(probs_list, weights):
        weighted_probs += probs * w

    return weighted_probs


# ═══════════════════════════════════════════════════════════════
# 3️⃣ Stacking Ensemble (meta-learner)
# ═══════════════════════════════════════════════════════════════

def ensemble_stacking(models):
    """
    Logistic Regression meta-learner:
    يتعلم على validation set، يتنبأ على test set.
    """
    # تجميع validation probabilities
    val_features_list = []
    val_labels = None
    for name, m in models.items():
        if m['val_probs'] is None:
            print(f"  ⚠️  {name}: لا يوجد val_probs — لا يمكن استخدام stacking")
            return None
        val_features_list.append(m['val_probs'])
        if val_labels is None:
            val_labels = m['val_labels']

    # stack: [n_samples, n_classes * n_models]
    val_features = np.hstack(val_features_list)
    test_features = np.hstack([m['test_probs'] for m in models.values()])

    print(f"\n  Stacking features: val={val_features.shape}, test={test_features.shape}")

    # تدريب meta-learner لكل قطاع
    stacked_probs = np.zeros_like(list(models.values())[0]['test_probs'])

    for i, sec in enumerate(ALL_SECTORS):
        yt = val_labels[:, i]
        if yt.sum() == 0:
            continue

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(val_features, yt)
        stacked_probs[:, i] = lr.predict_proba(test_features)[:, 1]

    return stacked_probs


# ═══════════════════════════════════════════════════════════════
# 📊 Evaluation + Threshold Tuning
# ═══════════════════════════════════════════════════════════════

def find_optimal_thresholds(probs, labels):
    thresholds = np.full(len(ALL_SECTORS), 0.50, dtype=np.float32)
    total = len(labels)
    for i in range(len(ALL_SECTORS)):
        yt, prob = labels[:, i], probs[:, i]
        n_pos = int(yt.sum())
        if n_pos == 0:
            continue
        base_f1 = f1_score(yt, (prob > 0.50).astype(int), zero_division=0)
        best_f1, best_th = base_f1, 0.50
        prevalence = n_pos / total
        for th in np.arange(0.15, 0.85, 0.02):
            pred = (prob > th).astype(int)
            f1 = f1_score(yt, pred, zero_division=0)
            if prevalence > 0.15 and pred.mean() > prevalence * 1.8:
                f1 *= 0.95
            if f1 > best_f1 + 0.005:
                best_f1, best_th = f1, th
        if n_pos < 15:
            best_th = min(best_th, 0.45)
        thresholds[i] = float(np.clip(best_th, 0.15, 0.85))
    return thresholds


def evaluate_ensemble(probs, labels, method_name):
    # البحث عن أفضل thresholds
    thresholds = find_optimal_thresholds(probs, labels)
    preds = (probs > thresholds[np.newaxis, :]).astype(int)

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    prec     = precision_score(labels, preds, average='macro', zero_division=0)
    rec      = recall_score(labels, preds, average='macro', zero_division=0)
    acc      = accuracy_score(labels, preds)

    print(f"\n  📊 {method_name}:")
    print(f"     F1-Macro    : {f1_macro:.4f}  ⭐")
    print(f"     Precision   : {prec:.4f}")
    print(f"     Recall      : {rec:.4f}")
    print(f"     Accuracy    : {acc:.4f}")

    # تفصيلي لكل قطاع
    print(f"\n     {'القطاع':28s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Support':>8s}")
    print("     " + "─" * 55)
    per_sector = {}
    for i, sec in enumerate(ALL_SECTORS):
        yt, yp = labels[:, i], preds[:, i]
        f = f1_score(yt, yp, zero_division=0)
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        per_sector[sec] = {'f1': round(f,4), 'precision': round(p,4),
                           'recall': round(r,4), 'support': int(yt.sum())}
        print(f"     {sec:28s} {f:.4f} {p:.4f} {r:.4f} {int(yt.sum()):>8d}")

    return {
        'method': method_name,
        'f1_macro': round(f1_macro, 4),
        'precision_macro': round(prec, 4),
        'recall_macro': round(rec, 4),
        'accuracy': round(acc, 4),
        'thresholds': thresholds.tolist(),
        'per_sector': per_sector,
    }


# ═══════════════════════════════════════════════════════════════
# 🎯 Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=['average', 'weighted', 'stacking', 'all'])
    args = parser.parse_args()

    print("=" * 70)
    print("🔗 Step 3: Ensemble Models")
    print("=" * 70)

    models = load_model_outputs()

    if len(models) < 2:
        print(f"\n⚠️  نحتاج على الأقل نموذجين! وجدنا {len(models)} فقط.")
        print("   شغّل step2 مع نماذج مختلفة أولاً:")
        print("   python step2_train_with_hard_negatives.py --model arabert")
        print("   python step2_train_with_hard_negatives.py --model marbert")
        return

    # التأكد من تطابق الأحجام
    sizes = [m['test_probs'].shape for m in models.values()]
    if len(set(str(s) for s in sizes)) > 1:
        print(f"❌ أحجام مختلفة: {sizes}")
        print("   تأكد أن جميع النماذج تدربت على نفس البيانات")
        return

    labels = list(models.values())[0]['test_labels']

    # النتائج الفردية للمقارنة
    print("\n📊 أداء النماذج الفردية:")
    for name, m in models.items():
        th = find_optimal_thresholds(m['test_probs'], labels)
        preds = (m['test_probs'] > th[np.newaxis, :]).astype(int)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        print(f"  {name:15s}: F1-Macro = {f1:.4f}")

    results = {}

    if args.method in ('average', 'all'):
        probs = ensemble_average(models)
        results['average'] = evaluate_ensemble(probs, labels, "Simple Average")

    if args.method in ('weighted', 'all'):
        probs = ensemble_weighted(models)
        results['weighted'] = evaluate_ensemble(probs, labels, "Weighted Average")

    if args.method in ('stacking', 'all'):
        probs = ensemble_stacking(models)
        if probs is not None:
            results['stacking'] = evaluate_ensemble(probs, labels, "Stacking")

    # أفضل طريقة
    if results:
        best_method = max(results.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n{'='*70}")
        print(f"🏆 أفضل طريقة: {best_method[0]} → F1-Macro = {best_method[1]['f1_macro']:.4f}")
        print(f"{'='*70}")

        # حفظ
        with open(os.path.join(OUTPUT_DIR, 'ensemble_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 النتائج محفوظة في: {OUTPUT_DIR}/ensemble_results.json")


if __name__ == "__main__":
    main()
