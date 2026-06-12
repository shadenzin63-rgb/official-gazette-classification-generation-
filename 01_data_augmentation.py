# -*- coding: utf-8 -*-
"""
الخطوة 1: Data Augmentation للقطاعات النادرة (نسخة محسّنة)
════════════════════════════════════════════════════════════
✅ إصلاح: إزالة Stop Words من جميع النصوص (أصلية ومولدة)
✅ إضافة: دالة post_process() للتنظيف النهائي
✅ تحسين: حفظ نسخة نظيفة من البيانات
"""

import os
import re
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# ⚙️ الإعدادات
# ═══════════════════════════════════════════════════════════════

INPUT_FILE  = "/home/albara/Desktop/project NLP/data/Final_Merged_to_classification.xlsx"
OUTPUT_FILE = "/home/albara/Desktop/project NLP/data/Augmented_Data_2_CLEAN.xlsx"

COLUMN_FILE_NAME = "اسم الملف"
COLUMN_SECTORS   = "القطاعات"
COLUMN_TEXT      = "text"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# القطاعات
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

TARGET_MIN_SAMPLES = 120
EDA_COPIES_PER_DOC = 3

# ═══════════════════════════════════════════════════════════════
# 🔧 كلمات الوقف والمرادفات
# ═══════════════════════════════════════════════════════════════

ARABIC_STOP_WORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'التي', 'الذي', 'اللذان', 'اللتان', 'الذين', 'اللاتي',
    'أن', 'إن', 'لا', 'ما', 'لم', 'لن', 'قد', 'كان', 'يكون', 'هو',
    'هي', 'هم', 'هن', 'نحن', 'أنا', 'أنت', 'أنتم', 'بين', 'كل',
    'أو', 'و', 'ثم', 'حتى', 'إذا', 'إذ', 'لأن', 'لكن', 'بل',
    'المادة', 'الفقرة', 'البند', 'القانون', 'النظام', 'القرار',
    'عدد', 'العدد', 'الرقم', 'رقم', 'الماده', 'مادة', 'ماده', 'الى',
}

LEGAL_SYNONYMS = {
    'قانون':     ['تشريع', 'نظام', 'قرار'],
    'تشريع':     ['قانون', 'نظام', 'تنظيم'],
    'نظام':      ['قانون', 'تشريع', 'لائحة'],
    'قرار':      ['مرسوم', 'أمر', 'توجيه'],
    'مرسوم':     ['قرار', 'أمر'],
    'لائحة':     ['تعليمات', 'نظام'],
    'تعليمات':   ['لائحة', 'إرشادات', 'أحكام'],
    'محكمة':     ['قضاء', 'هيئة قضائية'],
    'عقوبة':     ['جزاء', 'حكم'],
    'جزاء':      ['عقوبة', 'غرامة'],
    'غرامة':     ['جزاء مالي', 'غرامة مالية'],
    'ضريبة':     ['رسم', 'عوائد'],
    'رسم':       ['ضريبة', 'أجر'],
    'موظف':      ['عامل', 'مستخدم'],
    'عامل':      ['موظف', 'مستخدم', 'أجير'],
    'راتب':      ['أجر', 'مكافأة'],
    'أجر':       ['راتب', 'تعويض'],
    'شركة':      ['مؤسسة', 'منشأة'],
    'مؤسسة':     ['شركة', 'هيئة', 'منشأة'],
    'أرض':       ['عقار', 'ملك'],
    'عقار':      ['أرض', 'ملك', 'مسكن'],
    'تعليم':     ['تدريس', 'تربية'],
    'صحة':       ['رعاية صحية', 'طب'],
    'بيئة':      ['محيط', 'طبيعة'],
    'نقل':       ['مواصلات', 'سير'],
    'سياحة':     ['ترفيه', 'سفر'],
    'أمن':       ['حماية', 'دفاع'],
    'إعلام':     ['صحافة', 'نشر'],
    'يجوز':      ['يحق', 'يُسمح'],
    'يحظر':      ['يُمنع', 'لا يجوز'],
    'يُعاقب':    ['يُجازى', 'يُغرّم'],
    'ينشر':      ['يُعلن', 'يُصدر'],
    'يُلغى':     ['يُبطل', 'يُنسخ'],
    'وفقاً':     ['طبقاً', 'بموجب', 'استناداً'],
    'بموجب':     ['وفقاً', 'طبقاً', 'بمقتضى'],
}

# ═══════════════════════════════════════════════════════════════
# ✅ دوال التنظيف المحسّنة
# ═══════════════════════════════════════════════════════════════

def remove_stop_words(text: str) -> str:
    """إزالة كلمات الوقف فقط"""
    words = text.split()
    words = [w for w in words if w not in ARABIC_STOP_WORDS]
    return ' '.join(words)

def clean_arabic_text(text: str, remove_stops: bool = True) -> str:
    """
    تنظيف شامل للنص العربي
    
    Args:
        text: النص المراد تنظيفه
        remove_stops: إذا True، يحذف stop words
    """
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
        return ""
    
    # 1️⃣ إزالة الروابط
    text = re.sub(r'http\S+|www\S+|mailto:\S+', '', text)
    
    # 2️⃣ إزالة التشكيل
    text = re.sub(r'[\u064B-\u0652\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
    
    # 3️⃣ إزالة الأرقام والرموز الإنجليزية
    text = re.sub(r'[0-9a-zA-Z_]+', '', text)
    
    # 4️⃣ إزالة الكلمات الشكلية
    text = re.sub(r'\b(عدد|العدد|الرقم|رقم|المادة|مادة|ماده|الى|إلى)\b', ' ', text)
    
    # 5️⃣ تنظيف المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6️⃣ 🟢 إزالة Stop Words
    if remove_stops:
        text = remove_stop_words(text)
    
    return text

def post_process_text(text: str) -> str:
    """
    معالجة نهائية للنص بعد Augmentation
    - تضمن أن جميع stop words محذوفة
    - تنظف المسافات الزائدة
    """
    if not text or len(text.strip()) < 10:
        return ""
    
    # إزالة stop words مرة أخرى (للضمان)
    text = remove_stop_words(text)
    
    # تنظيف نهائي
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if len(text) > 10 else ""

# ═══════════════════════════════════════════════════════════════
# 1️⃣  EDA: Easy Data Augmentation
# ═══════════════════════════════════════════════════════════════

def eda_synonym_replace(words: list, n: int = 2) -> list:
    """استبدال n كلمات بمرادفات قانونية"""
    new_words = words.copy()
    replaceable = [i for i, w in enumerate(words) if w in LEGAL_SYNONYMS]
    random.shuffle(replaceable)
    for idx in replaceable[:n]:
        synonym = random.choice(LEGAL_SYNONYMS[words[idx]])
        new_words[idx] = synonym
    return new_words

def eda_random_delete(words: list, p: float = 0.1) -> list:
    """حذف عشوائي بنسبة p (لا يحذف stop words)"""
    if len(words) <= 5:
        return words
    new_words = [w for w in words if random.random() > p]
    return new_words if len(new_words) > 3 else words

def eda_random_swap(words: list, n: int = 2) -> list:
    """تبديل مواقع n أزواج من الكلمات"""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        i, j = random.sample(range(len(new_words)), 2)
        new_words[i], new_words[j] = new_words[j], new_words[i]
    return new_words

def eda_random_insert(words: list, n: int = 1) -> list:
    """إدخال n كلمات من المرادفات في مواقع عشوائية"""
    new_words = words.copy()
    all_synonyms = [s for syns in LEGAL_SYNONYMS.values() for s in syns]
    for _ in range(n):
        pos = random.randint(0, len(new_words))
        new_words.insert(pos, random.choice(all_synonyms))
    return new_words

def augment_text_eda(text: str, num_copies: int = 3) -> list:
    """
    توليد نسخ معدلة بـ EDA
    ✅ ينظف النص قبل وبعد التوليد
    """
    # تنظيف أولي
    text = clean_arabic_text(text, remove_stops=True)
    
    if not text or len(text.strip()) < 20:
        return []
    
    words = text.strip().split()
    augmented = []
    
    techniques = [
        lambda w: eda_synonym_replace(w, n=max(1, len(w) // 20)),
        lambda w: eda_random_delete(w, p=0.08),
        lambda w: eda_random_swap(w, n=max(1, len(w) // 30)),
        lambda w: eda_random_insert(w, n=max(1, len(w) // 40)),
        lambda w: eda_random_delete(eda_synonym_replace(w, n=2), p=0.05),
        lambda w: eda_random_swap(eda_synonym_replace(w, n=1), n=1),
    ]
    
    for i in range(num_copies):
        technique = techniques[i % len(techniques)]
        new_words = technique(words)
        new_text = ' '.join(new_words)
        
        # 🟢 معالجة نهائية
        new_text = post_process_text(new_text)
        
        if new_text and new_text != text and len(new_text) > 20:
            augmented.append(new_text)
    
    return augmented

# ═══════════════════════════════════════════════════════════════
# 2️⃣  Contextual Augmentation باستخدام AraBERT MLM
# ═══════════════════════════════════════════════════════════════

def try_contextual_augmentation():
    """محاولة تحميل AraBERT للـ masked language model"""
    try:
        from transformers import pipeline
        fill_mask = pipeline(
            'fill-mask',
            model='aubmindlab/bert-base-arabertv2',
            top_k=3,
            device=-1
        )
        print("  ✅ AraBERT MLM محمّل للـ Contextual Augmentation")
        return fill_mask
    except Exception as e:
        print(f"  ⚠️  لم يتم تحميل MLM: {e}")
        print("  → سيتم استخدام EDA فقط")
        return None

def augment_text_contextual(text: str, fill_mask, num_copies: int = 2) -> list:
    """
    توليد نسخ معدلة باستخدام AraBERT MLM
    ✅ ينظف النص قبل وبعد التوليد
    """
    # تنظيف أولي
    text = clean_arabic_text(text, remove_stops=True)
    
    if not text or len(text.strip()) < 30:
        return []
    
    words = text.strip().split()
    if len(words) < 5:
        return []
    
    augmented = []
    
    for _ in range(num_copies):
        new_words = words.copy()
        
        # اختيار كلمة للاستبدال
        maskable = [i for i, w in enumerate(words) 
                   if len(w) > 3 and w not in ARABIC_STOP_WORDS]
        
        if not maskable:
            continue
        
        mask_idx = random.choice(maskable)
        masked_text = ' '.join(new_words[:mask_idx] + ['[MASK]'] + new_words[mask_idx+1:])
        
        try:
            predictions = fill_mask(masked_text)
            if predictions and len(predictions) > 0:
                replacement = predictions[0]['token_str'].strip()
                new_words[mask_idx] = replacement
        except:
            continue
        
        new_text = ' '.join(new_words)
        
        # 🟢 معالجة نهائية
        new_text = post_process_text(new_text)
        
        if new_text and new_text != text:
            augmented.append(new_text)
    
    return augmented

# ═══════════════════════════════════════════════════════════════
# 3️⃣  Sector-Aware Functions
# ═══════════════════════════════════════════════════════════════

def parse_sectors(s: str) -> list:
    """تحليل القطاعات مع الدمج"""
    if pd.isna(s) or not isinstance(s, str):
        return []
    sectors = [p.strip() for p in re.split(r'\s*[|¦,;/]\s*', s.strip()) if p.strip()]
    merged = []
    for sector in sectors:
        if sector in SECTOR_MERGE_MAP:
            target = SECTOR_MERGE_MAP[sector]
            if target in SECTOR_TO_ID:
                merged.append(target)
        elif sector in SECTOR_TO_ID:
            merged.append(sector)
    return list(set(merged))

def count_sectors(df: pd.DataFrame) -> dict:
    """عدّ الوثائق في كل قطاع"""
    counts = {s: 0 for s in ALL_SECTORS}
    for _, row in df.iterrows():
        for sector in parse_sectors(str(row[COLUMN_SECTORS])):
            counts[sector] += 1
    return counts

def get_rare_sectors(counts: dict, threshold: int = None) -> list:
    """القطاعات التي عددها أقل من threshold"""
    if threshold is None:
        threshold = TARGET_MIN_SAMPLES
    return [s for s, c in counts.items() if c < threshold]

# ═══════════════════════════════════════════════════════════════
# 4️⃣  التنفيذ الرئيسي مع التنظيف النهائي
# ═══════════════════════════════════════════════════════════════

def augment_dataset(df: pd.DataFrame, fill_mask=None) -> pd.DataFrame:
    """
    توسيع البيانات مع ضمان تنظيف جميع النصوص
    """
    # 🟢 تنظيف النصوص الأصلية أولاً
    print("\n🧹 تنظيف النصوص الأصلية...")
    df[COLUMN_TEXT] = df[COLUMN_TEXT].apply(
        lambda x: clean_arabic_text(str(x), remove_stops=True)
    )
    df = df[df[COLUMN_TEXT].str.len() >= 20].reset_index(drop=True)
    print(f"  ✅ تم تنظيف {len(df)} نص")
    
    counts = count_sectors(df)
    rare_sectors = get_rare_sectors(counts)
    
    print(f"\n📊 التوزيع الحالي:")
    print(f"  {'القطاع':28s} {'عدد':>6s} {'الحالة':>10s}")
    print("  " + "─" * 50)
    for sec in ALL_SECTORS:
        status = "⚠️  نادر" if sec in rare_sectors else "✅ كافي"
        print(f"  {sec:28s} {counts[sec]:>6d} {status:>10s}")
    
    if not rare_sectors:
        print("\n✅ لا توجد قطاعات نادرة!")
        return df
    
    print(f"\n🔄 القطاعات المستهدفة للتوسيع ({len(rare_sectors)}):")
    for sec in rare_sectors:
        needed = TARGET_MIN_SAMPLES - counts[sec]
        print(f"  • {sec}: {counts[sec]} → {TARGET_MIN_SAMPLES} (نحتاج +{needed})")
    
    # توليد عينات جديدة
    new_rows = []
    for sec in tqdm(rare_sectors, desc="Augmenting sectors"):
        sector_docs = []
        for idx, row in df.iterrows():
            secs = parse_sectors(str(row[COLUMN_SECTORS]))
            if sec in secs:
                sector_docs.append(row)
        
        current = len(sector_docs)
        needed = TARGET_MIN_SAMPLES - current
        if needed <= 0:
            continue
        
        copies_per_doc = max(1, needed // current + 1)
        generated = 0
        
        for doc in sector_docs:
            if generated >= needed:
                break
            
            text = str(doc[COLUMN_TEXT])
            
            # EDA augmentation
            eda_texts = augment_text_eda(text, num_copies=min(copies_per_doc, EDA_COPIES_PER_DOC))
            
            # Contextual augmentation
            ctx_texts = []
            if fill_mask is not None:
                ctx_texts = augment_text_contextual(
                    text, fill_mask, num_copies=min(2, copies_per_doc))
            
            all_aug = eda_texts + ctx_texts
            
            for aug_text in all_aug:
                if generated >= needed:
                    break
                new_row = doc.copy()
                new_row[COLUMN_TEXT] = aug_text
                new_row[COLUMN_FILE_NAME] = f"AUG_{sec[:8]}_{generated}_{doc[COLUMN_FILE_NAME]}"
                new_rows.append(new_row)
                generated += 1
        
        print(f"  ✅ {sec}: +{generated} عينة جديدة (المجموع: {current + generated})")
    
    if not new_rows:
        print("\n⚠️  لم يتم توليد أي عينات!")
        return df
    
    # دمج البيانات
    aug_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    # 🟢 تنظيف نهائي لجميع النصوص
    print("\n🧹 تنظيف نهائي لجميع النصوص...")
    final_df[COLUMN_TEXT] = final_df[COLUMN_TEXT].apply(post_process_text)
    final_df = final_df[final_df[COLUMN_TEXT].str.len() >= 20].reset_index(drop=True)
    
    # طباعة التوزيع الجديد
    new_counts = count_sectors(final_df)
    print(f"\n📊 التوزيع بعد التوسيع:")
    print(f"  {'القطاع':28s} {'قبل':>6s} {'بعد':>6s} {'زيادة':>8s}")
    print("  " + "─" * 55)
    for sec in ALL_SECTORS:
        diff = new_counts[sec] - counts[sec]
        mark = f"+{diff}" if diff > 0 else "—"
        print(f"  {sec:28s} {counts[sec]:>6d} {new_counts[sec]:>6d} {mark:>8s}")
    
    print(f"\n  📦 الإجمالي: {len(df)} → {len(final_df)} (+{len(final_df)-len(df)})")
    return final_df

# ═══════════════════════════════════════════════════════════════
# 5️⃣  Downsampling للتشريعات
# ═══════════════════════════════════════════════════════════════

def downsample_dominant_sector(df: pd.DataFrame,
                                sector: str = "تشريعات وقرارات عليا",
                                max_ratio: float = 0.25) -> pd.DataFrame:
    """تقليل عينات قطاع التشريعات لخفض هيمنته"""
    total = len(df)
    max_count = int(total * max_ratio)
    
    legislation_only_indices = []
    legislation_multi_indices = []
    
    for idx, row in df.iterrows():
        secs = parse_sectors(str(row[COLUMN_SECTORS]))
        if sector in secs:
            if len(secs) == 1:
                legislation_only_indices.append(idx)
            else:
                legislation_multi_indices.append(idx)
    
    current_count = len(legislation_only_indices) + len(legislation_multi_indices)
    
    if current_count <= max_count:
        print(f"  ℹ️  {sector} ({current_count}) أقل من الحد ({max_count}) — لا حاجة للتقليل")
        return df
    
    keep_single = max(0, max_count - len(legislation_multi_indices))
    if keep_single < len(legislation_only_indices):
        random.shuffle(legislation_only_indices)
        remove_indices = legislation_only_indices[keep_single:]
        df = df.drop(remove_indices).reset_index(drop=True)
        removed = len(remove_indices)
        print(f"  🔽 {sector}: تم تقليل {removed} عينة "
              f"({current_count} → {current_count - removed})")
    return df

# ═══════════════════════════════════════════════════════════════
# 🎯  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🔄 Step 1: Data Augmentation (نسخة محسّنة)")
    print("=" * 70)
    
    # تحميل البيانات
    print(f"\n📂 تحميل البيانات من: {INPUT_FILE}")
    try:
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
        print(f"  ✅ {len(df)} صف")
    except FileNotFoundError:
        print(f"❌ الملف غير موجود: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return
    
    # محاولة تحميل AraBERT MLM
    print("\n🧠 تحميل AraBERT MLM (اختياري)...")
    fill_mask = try_contextual_augmentation()
    
    # تقليل هيمنة التشريعات
    print("\n🔽 Downsampling للتشريعات...")
    df = downsample_dominant_sector(df, max_ratio=0.25)
    
    # التوسيع مع التنظيف
    print("\n🔄 بدء Data Augmentation...")
    augmented_df = augment_dataset(df, fill_mask)
    
    # خلط البيانات
    augmented_df = augmented_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # 🟢 فحص نهائي: عرض عينة من النصوص
    print("\n🔍 عينة من النصوص النهائية:")
    for i in range(min(3, len(augmented_df))):
        sample = augmented_df.iloc[i][COLUMN_TEXT][:100]
        has_stops = any(word in ARABIC_STOP_WORDS for word in sample.split())
        status = "❌ يحتوي stop words" if has_stops else "✅ نظيف"
        print(f"  {i+1}. {sample}... {status}")
    
    # حفظ
    print(f"\n💾 حفظ البيانات الموسّعة: {OUTPUT_FILE}")
    augmented_df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"  ✅ تم الحفظ: {len(augmented_df)} وثيقة")
    
    print(f"\n✅ انتهى! الملف الجديد: {OUTPUT_FILE}")
    print(f"   جميع النصوص نظيفة من Stop Words")

if __name__ == "__main__":
    main()