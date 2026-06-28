import re
import torch
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

threshold = 6
fold_out = 5
fold_in = 3
ridge_alpha = [0.1, 1.0, 10.0, 100.0]
enseble = (0.4, 0.6)
questions = [("q1", "score1", "Q1: Почему именно вы?"),("q2", "score2", "Q2: Применение знаний"),("q3", "score3", "Q3: Задача сверх обязанностей"),]

df = pd.read_excel("tests.xlsx", engine="openpyxl").replace(r"^\s*([.\-*xX]|норм|Норм|НОРМ)\s*$", np.nan, regex=True)
df.columns = ["id", "q1", "score1", "q2", "score2", "q3", "score3", "total"]
df = df.dropna(thresh=2)

train = df[df["id"].str.startswith("train")].copy().reset_index(drop=True)
val = df[df["id"].str.startswith("val")].copy().reset_index(drop=True)
val["id"] = [f"val_{i}" for i in range(len(val))]
print(f"Train: {len(train)}, Val: {len(val)}")

def clean_text(text) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for q_col in ["q1", "q2", "q3"]:
        df[f"is_missing_{q_col}"] = df[q_col].isna().astype(int)
        df[q_col] = df[q_col].apply(clean_text)
    return df

train = preprocess(train)
val = preprocess(val)

def text_features(texts: pd.Series, is_missing_flags: pd.Series) -> np.ndarray:
    feats = []
    for t, missing in zip(texts, is_missing_flags):
        words = re.findall(r"[а-яёa-z]+", t.lower())
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        n_w = len(words)
        n_u = len(set(words))
        n_s = max(len(sents), 1)
        feats.append([
            len(t),
            n_w,
            n_u,
            n_s,
            int(bool(re.search(r"\d+", t))),
            np.mean([len(w) for w in words]) if words else 0,
            n_u / n_w if n_w > 0 else 0,
            n_w / n_s,
            int(missing),
        ])
    return np.array(feats, dtype=np.float32)

def build_tfidf(texts: pd.Series):
    char_v = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), max_features=2000, sublinear_tf=True
    )
    word_v = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), max_features=2000,
        sublinear_tf=True, min_df=2
    )
    char_v.fit(texts)
    word_v.fit(texts)
    return char_v, word_v

def encode(texts: pd.Series, missing_flags: pd.Series, char_v, word_v) -> np.ndarray:
    return np.hstack([
        char_v.transform(texts).toarray(),
        word_v.transform(texts).toarray(),
        text_features(texts, missing_flags),
    ])
    
def qwk(y_true: np.ndarray, y_pred: np.ndarray, max_score: int = 3) -> float:
    y_r = np.round(y_true).astype(int).clip(0, max_score)
    y_p = np.round(y_pred).astype(int).clip(0, max_score)
    try:
        return cohen_kappa_score(y_r, y_p, weights="quadratic")
    except Exception:
        return float("nan")

models_final = {}
tfidf_store = {}
all_oof = np.zeros(len(train))
fold_outf = KFold(n_splits=fold_out, shuffle=True, random_state=42)

for q_col, s_col, q_name in questions:
    y = train[s_col].fillna(0).values
    miss_flags = train[f"is_missing_{q_col}"]
    texts = train[q_col]
    oof = np.zeros(len(y))
    best_alphas_list = []
    for fold, (tr_i, val_i) in enumerate(fold_outf.split(texts)):
        char_v_fold, word_v_fold = build_tfidf(texts.iloc[tr_i])
        X_tr = encode(texts.iloc[tr_i], miss_flags.iloc[tr_i], char_v_fold, word_v_fold)
        X_val = encode(texts.iloc[val_i], miss_flags.iloc[val_i], char_v_fold, word_v_fold)
        y_tr, y_val = y[tr_i], y[val_i]

        fold_inf = KFold(n_splits=fold_in, shuffle=True, random_state=fold)
        best_alpha, best_mae_inner = ridge_alpha[1], 999.0
        for alpha in ridge_alpha:
            inner_maes = []
            for ti, vi in fold_inf.split(X_tr):
                r = Ridge(alpha=alpha)
                r.fit(X_tr[ti], y_tr[ti])
                p = np.clip(r.predict(X_tr[vi]), 0, 3)
                inner_maes.append(mean_absolute_error(y_tr[vi], p))
            if np.mean(inner_maes) < best_mae_inner:
                best_mae_inner = np.mean(inner_maes)
                best_alpha = alpha
            best_alphas_list.append(best_alpha)
            ridge = Ridge(alpha=best_alpha)
        final_alpha = np.mean(best_alphas_list)
        ridge_f = Ridge(alpha=final_alpha)
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=42,
        )
        ridge.fit(X_tr, y_tr)
        gbm.fit(X_tr, y_tr)
        preds = np.clip(enseble[0] * ridge.predict(X_val) + enseble[1] * gbm.predict(X_val),0,3,)
        oof[val_i] = preds

    char_v_f, word_v_f = build_tfidf(texts)
    X_full = encode(texts, miss_flags, char_v_f, word_v_f)
    ridge_f = Ridge(alpha=1.0)
    gbm_f = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    ridge_f.fit(X_full, y)
    gbm_f.fit(X_full, y)

    models_final[q_col] = (ridge_f, gbm_f)
    tfidf_store[q_col] = (char_v_f, word_v_f)
    all_oof += oof

    print(f"{q_name} MAE: {mean_absolute_error(y, oof):.3f}, QWK: {qwk(y, oof, 3):.3f}")

total_true = train["total"].fillna(0).values
pt = (total_true >= threshold).astype(int)
pp = (all_oof >= threshold).astype(int)

print(
    f"MAE: {mean_absolute_error(total_true, all_oof):.3f}, "
    f"RMSE: {mean_squared_error(total_true, all_oof) ** 0.5:.3f}, "
    f"QWK: {qwk(total_true, all_oof, 9):.3f}, "
    f"Acc: {accuracy_score(pt, pp):.3f}, "
    f"F1: {f1_score(pt, pp, zero_division=0):.3f}"
)

model_bert = "cointegrated/rubert-tiny2" # или используем "DeepPavlov/rubert-base-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_bert_embeddings(texts: list, model_name: str, device, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embeddings = []
    pad_str = tokenizer.pad_token or "[PAD]"
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [str(t) if str(t).strip() else pad_str for t in batch]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        all_embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeddings)

def encode_all_questions_bert(df: pd.DataFrame, model_name: str, device) -> dict:
    q_embeddings = {}
    for q_col in ["q1", "q2", "q3"]:
        q_embeddings[q_col] = get_bert_embeddings(df[q_col].tolist(), model_name, device)
    return q_embeddings

X_bert_train = encode_all_questions_bert(train, model_bert, device)
X_bert_val = encode_all_questions_bert(val, model_bert, device)

kf_bert = KFold(n_splits=5, shuffle=True, random_state=42)
bert_oof_total = np.zeros(len(train))
model_berts = {}

for q_col, s_col, q_name in questions:
    y = train[s_col].fillna(0).values
    oof = np.zeros(len(y))
    best_alphas_list = []
    for fold, (tr_i, val_i) in enumerate(kf_bert.split(X_bert_train[q_col])):
        fold_inf = KFold(n_splits=3, shuffle=True, random_state=fold)
        best_alpha, best_mae_in = 1.0, 999.0
        for alpha in ridge_alpha:
            inner_maes = []
            for ti, vi in fold_inf.split(X_bert_train[tr_i]):
                r = Ridge(alpha=alpha)
                r.fit(X_bert_train[tr_i][ti], y[tr_i][ti])
                p = np.clip(r.predict(X_bert_train[tr_i][vi]), 0, 3)
                inner_maes.append(mean_absolute_error(y[tr_i][vi], p))
            if np.mean(inner_maes) < best_mae_in:
                best_mae_in, best_alpha = np.mean(inner_maes), alpha
        best_alphas_list.append(best_alpha)
        r = Ridge(alpha=best_alpha)
        r.fit(X_bert_train[tr_i], y[tr_i])
        oof[val_i] = np.clip(r.predict(X_bert_train[val_i]), 0, 3)
    final_alpha = np.mean(best_alphas_list)
    final_r = Ridge(alpha=final_alpha)
    final_r.fit(X_bert_train, y)
    model_berts[q_col] = final_r
    bert_oof_total += oof
    print(f"{q_name} - MAE: {mean_absolute_error(y, oof):.3f}  QWK: {qwk(y, oof, 3):.3f}")

total_true = train["total"].fillna(0).values
print(f"MAE: {mean_absolute_error(total_true, bert_oof_total):.3f}  QWK: {qwk(total_true, bert_oof_total, 9):.3f}")

bert_val_preds = {}
for q_col, s_col, _ in questions:
    bert_val_preds[s_col] = np.clip(model_berts[q_col].predict(X_bert_val[q_col]), 0, 3)

predictions = pd.DataFrame(
    {
        "id": val["id"],
        "pred_score1": np.round(bert_val_preds["score1"], 2),
        "pred_score2": np.round(bert_val_preds["score2"], 2),
        "pred_score3": np.round(bert_val_preds["score3"], 2),
    }
)
predictions["pred_total"] = (predictions["pred_score1"]+ predictions["pred_score2"]+ predictions["pred_score3"]).round(2)
predictions["pass"] = (predictions["pred_total"] >= threshold).astype(int)
predictions.to_excel("val_predictions.xlsx", index=False)
print(f"Прошло порог (>= {threshold}): {predictions['pass'].mean():.1%} ({predictions['pass'].sum()} из {len(predictions)})")
