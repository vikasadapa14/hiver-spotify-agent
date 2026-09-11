import os
import time
import importlib.util
from pathlib import Path
import pandas as pd
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load 03_agent.py dynamically
agent_path = Path(__file__).resolve().parent / "03_agent.py"
spec = importlib.util.spec_from_file_location("agent_module", agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

baseline_keyword_agent = agent_module.baseline_keyword_agent
llm_agent = agent_module.llm_agent
get_groq_client = agent_module.get_groq_client

nltk.download('punkt', quiet=True)

def compute_metrics(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    rouge1_f, rouge2_f, rougeL_f = [], [], []
    bleu_scores = []
    char_lengths = []
    under_280_count = 0
    cta_count = 0

    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip()
        ref_str = str(ref).strip()

        # Length constraint
        char_lengths.append(len(pred_str))
        if len(pred_str) <= 280:
            under_280_count += 1

        # Call-to-action presence
        lower_pred = pred_str.lower()
        if any(k in lower_pred for k in ["dm", "direct message", "email", "settings", "reinstall", "spoti.fi"]):
            cta_count += 1

        # ROUGE Scores
        scores = scorer.score(ref_str, pred_str)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)

        # BLEU Score
        ref_tokens = ref_str.lower().split()
        pred_tokens = pred_str.lower().split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

    n = max(len(predictions), 1)
    return {
        "Avg Char Length": round(sum(char_lengths) / n, 2),
        "% Under 280 Chars": round((under_280_count / n) * 100, 2),
        "% With CTA / Action": round((cta_count / n) * 100, 2),
        "ROUGE-1 (F1)": round(sum(rouge1_f) / n, 4),
        "ROUGE-2 (F1)": round(sum(rouge2_f) / n, 4),
        "ROUGE-L (F1)": round(sum(rougeL_f) / n, 4),
        "BLEU Score": round(sum(bleu_scores) / n, 4),
    }

def run_evaluation(golden_path="data/golden_set_150.csv", sample_per_intent=5):
    print(f"Loading golden set from {golden_path}...")
    df = pd.read_csv(golden_path)

    # 1. Evaluate Rule Baseline on all 150 items
    print(f"Evaluating Baseline Rule Agent on all {len(df)} samples...")
    df["baseline_pred"] = df.apply(lambda r: baseline_keyword_agent(r["user_text"], r["intent"]), axis=1)
    baseline_metrics = compute_metrics(df["baseline_pred"], df["spotify_response"])

    # 2. Select 5 balanced samples per intent category
    sampled_frames = []
    for intent_name in df["intent"].unique():
        subset = df[df["intent"] == intent_name]
        sampled_frames.append(subset.sample(n=min(len(subset), sample_per_intent), random_state=42))
    
    sample_eval = pd.concat(sampled_frames, ignore_index=True)
    print(f"\nEvaluating LLM Agent on {len(sample_eval)} balanced samples (5 per intent)...")

    client = get_groq_client()
    llm_preds = []

    for idx, row in sample_eval.iterrows():
        intent = row["intent"]
        user_msg = row["user_text"]
        print(f"[{idx+1}/{len(sample_eval)}] Intent: {intent} | Processing...")
        
        try:
            reply = llm_agent(user_msg, client=client)
            llm_preds.append(reply)
            time.sleep(1.2)  # Respect rate limit
        except Exception as e:
            print(f"  Warning on sample {idx+1}: {e}. Falling back to baseline.")
            llm_preds.append(baseline_keyword_agent(user_msg, intent))

    sample_eval["llm_pred"] = llm_preds
    llm_metrics = compute_metrics(sample_eval["llm_pred"], sample_eval["spotify_response"])

    # 3. Print side-by-side benchmark
    comparison_df = pd.DataFrame([baseline_metrics, llm_metrics], index=["Baseline (Keyword)", "LLM Agent (Groq)"])
    print("\n" + "=" * 60)
    print("EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    print(comparison_df.to_string())
    print("=" * 60)

    # Export results
    sample_eval.to_csv("data/evaluation_sample_predictions.csv", index=False)
    comparison_df.to_csv("data/evaluation_metrics.csv")
    print("\n[OK] Saved sample predictions -> data/evaluation_sample_predictions.csv")
    print("[OK] Saved metrics summary -> data/evaluation_metrics.csv")

if __name__ == "__main__":
    run_evaluation()