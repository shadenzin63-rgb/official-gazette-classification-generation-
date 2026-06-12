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

# ── النماذج الأصلية الثلاثة ──────────────────────────────────
MAIN_MODEL_DIRS = {
    'arabert':   os.path.join(BASE_DIR, "v5_arabert_v2"),
    'marbert':   os.path.join(BASE_DIR, "v5_marbert"),
    'camelbert': os.path.join(BASE_DIR, "v5_camelbert_mix"),
}

# ── النموذج الرابع المتخصص ───────────────────────────────────
SPECIALIST_DIR = os.path.join(BASE_DIR, "v5_weak_specialist")

OUTPUT_DIR = os.path.join(BASE_DIR, "v5_ensemble_3")
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
    """تحميل probabilities وlabels من النماذج الثلاثة الأصلية"""
    models = {}
    for name, path in MAIN_MODEL_DIRS.items():
        test_probs_path  = os.path.join(path, 'test_probs.npy')
        test_labels_path = os.path.join(path, 'test_labels.npy')
        val_probs_path   = os.path.join(path, 'val_probs.npy')
        metrics_path     = os.path.join(path, 'final_metrics.json')

        if not os.path.exists(test_probs_path):
            print(f"  ⚠️  {name}: لم يتم تدريبه (تخطي)")
            continue

        test_probs  = np.load(test_probs_path)
        test_labels = np.load(test_labels_path)
        val_probs   = np.load(val_probs_path) if os.path.exists(val_probs_path) else None

        val_labels = None
        val_labels_path = os.path.join(path, 'val_labels.npy')
        if os.path.exists(val_labels_path):
            val_labels = np.load(val_labels_path)

        f1 = 0.0
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            f1 = data.get('test_metrics', {}).get('f1_macro', 0)

        models[name] = {
            'test_probs':  test_probs,
            'test_labels': test_labels,
            'val_probs':   val_probs,
            'val_labels':  val_labels,
            'f1_macro':    f1,
        }
        print(f"  ✅ {name:12s}: F1={f1:.4f}, shape={test_probs.shape}")

    return models


def load_specialist():
    """تحميل النموذج الرابع المتخصص"""
    test_probs_path = os.path.join(SPECIALIST_DIR, 'test_probs.npy')
    val_probs_path  = os.path.join(SPECIALIST_DIR, 'val_probs.npy')
    metrics_path    = os.path.join(SPECIALIST_DIR, 'final_metrics.json')

    if not os.path.exists(test_probs_path):
        print(f"  ⚠️  النموذج الرابع غير موجود في: {SPECIALIST_DIR}")
        return None

    test_probs = np.load(test_probs_path)
    val_probs  = np.load(val_probs_path) if os.path.exists(val_probs_path) else None

    f1 = 0.0
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as mf:
            data = json.load(mf)
        f1 = data.get('test_f1_specialist', 0)

    print(f"  ✅ {'specialist':12s}: F1={f1:.4f}, shape={test_probs.shape}")
    return {
        'test_probs': test_probs,
        'val_probs':  val_probs,
        'f1_macro':   f1,
    }


# ═══════════════════════════════════════════════════════════════
# 1️⃣ Simple Average Ensemble
# ═══════════════════════════════════════════════════════════════

def ensemble_average(models):
    probs_list = [m['test_probs'] for m in models.values()]
    
    # التحقق من تطابق الأحجام قبل المتوسط
    shapes = [p.shape for p in probs_list]
    if len(set(str(s) for s in shapes)) > 1:
        print(f"  ⚠️  أحجام مختلفة: {shapes}")
        # خذ الحجم الأصغر
        min_rows = min(p.shape[0] for p in probs_list)
        probs_list = [p[:min_rows] for p in probs_list]
        print(f"  → تم قص الكل إلى {min_rows} صف")
    
    return np.mean(probs_list, axis=0)


# ═══════════════════════════════════════════════════════════════
# 2️⃣ Weighted Average Ensemble
# ═══════════════════════════════════════════════════════════════

def ensemble_weighted(models):
    f1_scores  = np.array([m['f1_macro'] for m in models.values()])
    weights    = f1_scores ** 2
    weights    = weights / weights.sum()
    probs_list = [m['test_probs'] for m in models.values()]

    # توحيد الأحجام
    min_rows   = min(p.shape[0] for p in probs_list)
    probs_list = [p[:min_rows] for p in probs_list]

    weighted_probs = np.zeros_like(probs_list[0])
    for probs, w in zip(probs_list, weights):
        weighted_probs += probs * w
    return weighted_probs

# ═══════════════════════════════════════════════════════════════
# 4️⃣ Hybrid Ensemble (جديد)
# للتشريعات: النماذج الأصلية الثلاثة فقط
# لباقي القطاعات: الأربعة نماذج
# ═══════════════════════════════════════════════════════════════

# القطاعات التي يُستبعد منها المتخصص (مشكلة FP)
EXCLUDE_SPECIALIST_SECTORS = {
    "تشريعات وقرارات عليا",   # FP=74، downsampling أضعفها
}

def ensemble_hybrid(main_models, specialist, labels):
    """
    Hybrid Ensemble:
      - تشريعات وقرارات عليا → متوسط النماذج الثلاثة الأصلية فقط
      - باقي القطاعات          → متوسط الأربعة نماذج
    """
    if specialist is None:
        print("  ⚠️  المتخصص غير متاح → fallback لـ Average الأصلي")
        return ensemble_average(main_models)

    # توحيد الأحجام
    main_probs_list = [m['test_probs'] for m in main_models.values()]
    spec_probs      = specialist['test_probs']
    min_rows        = min(min(p.shape[0] for p in main_probs_list),
                          spec_probs.shape[0])
    main_probs_list = [p[:min_rows] for p in main_probs_list]
    spec_probs      = spec_probs[:min_rows]

    # متوسط النماذج الأصلية الثلاثة (لكل القطاعات)
    avg_main = np.mean(main_probs_list, axis=0)

    # متوسط الأربعة نماذج (لكل القطاعات)
    avg_all  = np.mean(main_probs_list + [spec_probs], axis=0)

    # دمج: لكل قطاع اختر المصدر الصح
    hybrid_probs = avg_all.copy()
    for i, sec in enumerate(ALL_SECTORS):
        if sec in EXCLUDE_SPECIALIST_SECTORS:
            hybrid_probs[:, i] = avg_main[:, i]

    excluded = list(EXCLUDE_SPECIALIST_SECTORS)
    included = [s for s in ALL_SECTORS if s not in EXCLUDE_SPECIALIST_SECTORS]
    print(f"\n  Hybrid Ensemble:")
    print(f"    3 نماذج فقط  ({len(excluded)} قطاع): {excluded}")
    print(f"    4 نماذج كاملة ({len(included)} قطاع): باقي القطاعات")

    return hybrid_probs


# ═══════════════════════════════════════════════════════════════
# 3️⃣ Stacking Ensemble (meta-learner)
# ═══════════════════════════════════════════════════════════════

def ensemble_stacking(models):
    val_features_list = []
    val_labels = None
    for name, m in models.items():
        if m['val_probs'] is None:
            print(f"  ⚠️  {name}: لا يوجد val_probs — لا يمكن استخدام stacking")
            return None
        val_features_list.append(m['val_probs'])
        if val_labels is None:
            val_labels = m['val_labels']

    if val_labels is None:
        print("  ⚠️  لا val_labels → تخطي Stacking")
        return None

    # توحيد أحجام val
    min_val_rows      = min(p.shape[0] for p in val_features_list)
    val_features_list = [p[:min_val_rows] for p in val_features_list]
    val_labels        = val_labels[:min_val_rows]

    val_features  = np.hstack(val_features_list)

    # توحيد أحجام test
    test_probs_list = [m['test_probs'] for m in models.values()]
    min_test_rows   = min(p.shape[0] for p in test_probs_list)
    test_probs_list = [p[:min_test_rows] for p in test_probs_list]

    test_features = np.hstack(test_probs_list)

    print(f"\n  Stacking features: val={val_features.shape}, test={test_features.shape}")

    stacked_probs = np.zeros((min_test_rows, len(ALL_SECTORS)), dtype=np.float32)

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
    print("🔗 Step 3: Ensemble - 3 تقييمات")
    print("=" * 70)

    # ── تحميل النماذج ───────────────────────────────────────
    print("\n📂 تحميل النماذج...")
    main_models = load_model_outputs()
    specialist  = load_specialist()

    if len(main_models) < 2:
        print(f"\n⚠️  نحتاج على الأقل نموذجين! وجدنا {len(main_models)} فقط.")
        return

    # الـ labels المرجعية من أول نموذج أصلي
    labels = list(main_models.values())[0]['test_labels']

    # التأكد من تطابق الأحجام
    for name, m in list(main_models.items()):
        if m['test_probs'].shape[0] != labels.shape[0]:
            print(f"  ⚠️  {name}: حجم مختلف → تخطي")
            del main_models[name]

    # ── اختيار أفضل نموذجين من الثلاثة ─────────────────────
    sorted_main = sorted(main_models.items(),
                         key=lambda x: x[1]['f1_macro'], reverse=True)
    top2_names  = [n for n, _ in sorted_main[:2]]
    top2_models = {n: main_models[n] for n in top2_names}

    print(f"\n  أفضل نموذجين F1   : {top2_names}")

    # أعلى نموذجين كفاءة = أعلى Precision
    def get_precision(m):
        th   = find_optimal_thresholds(m['test_probs'], labels)
        pred = (m['test_probs'] > th[np.newaxis, :]).astype(int)
        return precision_score(labels, pred, average='macro', zero_division=0)

    sorted_prec   = sorted(main_models.items(),
                           key=lambda x: get_precision(x[1]), reverse=True)
    top2p_names   = [n for n, _ in sorted_prec[:2]]
    top2p_models  = {n: main_models[n] for n in top2p_names}

    print(f"  أفضل نموذجين Prec  : {top2p_names}")

    all_results = {}

    # ══════════════════════════════════════════════════════════
    # التقييم 1: النماذج الثلاثة الرئيسية فقط
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("📊 التقييم 1: النماذج الثلاثة الرئيسية")
    print('='*70)

    results_1 = {}
    if args.method in ('average', 'all'):
        probs = ensemble_average(main_models)
        results_1['average'] = evaluate_ensemble(probs, labels, "Average")

    if args.method in ('weighted', 'all'):
        probs = ensemble_weighted(main_models)
        results_1['weighted'] = evaluate_ensemble(probs, labels, "Weighted")

    if args.method in ('stacking', 'all'):
        probs = ensemble_stacking(main_models)
        if probs is not None:
            results_1['stacking'] = evaluate_ensemble(probs, labels, "Stacking")

    if results_1:
        best_1 = max(results_1.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n  🏆 أفضل: {best_1[0]} → F1={best_1[1]['f1_macro']:.4f}")
    all_results['group_1_main_three'] = results_1

    # ══════════════════════════════════════════════════════════
    # التقييم 2: أعلى نموذجين F1 + النموذج الرابع
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"📊 التقييم 2: أعلى نموذجين F1 ({', '.join(top2_names)}) + المتخصص")
    print('='*70)

    results_2 = {}
    if specialist is not None:
        group2 = dict(top2_models)
        group2['specialist'] = {
            'test_probs':  specialist['test_probs'],
            'test_labels': labels,
            'val_probs':   specialist['val_probs'],
            'val_labels':  list(main_models.values())[0].get('val_labels'),
            'f1_macro':    specialist['f1_macro'],
        }

        if args.method in ('average', 'all'):
            probs = ensemble_average(group2)
            results_2['average'] = evaluate_ensemble(probs, labels, "Average")

        if args.method in ('weighted', 'all'):
            probs = ensemble_weighted(group2)
            results_2['weighted'] = evaluate_ensemble(probs, labels, "Weighted")

        if args.method in ('stacking', 'all'):
            probs = ensemble_stacking(group2)
            if probs is not None:
                results_2['stacking'] = evaluate_ensemble(probs, labels, "Stacking")

        if results_2:
            best_2 = max(results_2.items(), key=lambda x: x[1]['f1_macro'])
            print(f"\n  🏆 أفضل: {best_2[0]} → F1={best_2[1]['f1_macro']:.4f}")
    else:
        print("  ⚠️  النموذج الرابع غير متاح → تخطي")

    all_results['group_2_top2_f1_specialist'] = results_2

    # ══════════════════════════════════════════════════════════
    # التقييم 3: أعلى نموذجين Precision + النموذج الرابع
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"📊 التقييم 3: أعلى نموذجين Precision ({', '.join(top2p_names)}) + المتخصص")
    print('='*70)

    results_3 = {}
    if specialist is not None:
        group3 = dict(top2p_models)
        group3['specialist'] = {
            'test_probs':  specialist['test_probs'],
            'test_labels': labels,
            'val_probs':   specialist['val_probs'],
            'val_labels':  list(main_models.values())[0].get('val_labels'),
            'f1_macro':    specialist['f1_macro'],
        }

        if args.method in ('average', 'all'):
            probs = ensemble_average(group3)
            results_3['average'] = evaluate_ensemble(probs, labels, "Average")

        if args.method in ('weighted', 'all'):
            probs = ensemble_weighted(group3)
            results_3['weighted'] = evaluate_ensemble(probs, labels, "Weighted")

        if args.method in ('stacking', 'all'):
            probs = ensemble_stacking(group3)
            if probs is not None:
                results_3['stacking'] = evaluate_ensemble(probs, labels, "Stacking")

        if results_3:
            best_3 = max(results_3.items(), key=lambda x: x[1]['f1_macro'])
            print(f"\n  🏆 أفضل: {best_3[0]} → F1={best_3[1]['f1_macro']:.4f}")
    else:
        print("  ⚠️  النموذج الرابع غير متاح → تخطي")

    all_results['group_3_top2_precision_specialist'] = results_3

    # ══════════════════════════════════════════════════════════
    # التقييم 4: Hybrid (الثلاثة للتشريعات، الأربعة للباقي)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("📊 التقييم 4: Hybrid - الثلاثة للتشريعات، الأربعة للباقي")
    print('='*70)

    results_4 = {}
    if specialist is not None:
        probs = ensemble_hybrid(main_models, specialist, labels)
        min_rows = probs.shape[0]
        results_4['hybrid'] = evaluate_ensemble(
            probs, labels[:min_rows], "Hybrid")
        print(f"\n  🏆 F1={results_4['hybrid']['f1_macro']:.4f}")
    else:
        print("  ⚠️  النموذج الرابع غير متاح → تخطي")

    all_results['group_4_hybrid'] = results_4

    # ── ملخص المقارنة الكاملة ────────────────────────────────
    print(f"\n{'='*70}")
    print("📋 ملخص المقارنة بين التقييمات")
    print('='*70)
    print(f"  {'المجموعة':45s} {'أفضل طريقة':12s} {'F1-Macro':>10s}")
    print("  " + "─" * 70)

    group_labels = {
        'group_1_main_three':                '1: الثلاثة الرئيسيون',
        'group_2_top2_f1_specialist':        f'2: أعلى F1 ({", ".join(top2_names)}) + متخصص',
        'group_3_top2_precision_specialist': f'3: أعلى Prec ({", ".join(top2p_names)}) + متخصص',
        'group_4_hybrid':                    '4: Hybrid (تشريعات=3، باقي=4)',
    }

    overall_best_f1     = 0.0
    overall_best_label  = ""
    overall_best_method = ""

    for key, res in all_results.items():
        if not res:
            print(f"  {group_labels.get(key, key):45s} {'—':12s} {'N/A':>10s}")
            continue
        best  = max(res.items(), key=lambda x: x[1]['f1_macro'])
        label = group_labels.get(key, key)
        print(f"  {label:45s} {best[0]:12s} {best[1]['f1_macro']:>10.4f}")
        if best[1]['f1_macro'] > overall_best_f1:
            overall_best_f1     = best[1]['f1_macro']
            overall_best_label  = label
            overall_best_method = best[0]

    print(f"\n  🏆 الأفضل إجمالاً: {overall_best_label}")
    print(f"     الطريقة: {overall_best_method}  |  F1-Macro: {overall_best_f1:.4f}")
    print('='*70)

    # حفظ كل النتائج
    with open(os.path.join(OUTPUT_DIR, 'ensemble_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 النتائج محفوظة في: {OUTPUT_DIR}/ensemble_results.json")


if __name__ == "__main__":
    main()