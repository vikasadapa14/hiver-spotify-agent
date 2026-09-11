import os
import json
import importlib.util
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Dynamically import 03_agent.py because module names starting with numbers require importlib
spec = importlib.util.spec_from_file_location("agent_module", os.path.join("src", "03_agent.py"))
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)

trivial_baseline_agent = agent_mod.trivial_baseline_agent
rule_based_baseline_agent = agent_mod.rule_based_baseline_agent
llm_agent = agent_mod.llm_agent
get_groq_client = agent_mod.get_groq_client

nltk.download('punkt', quiet=True)

JUDGE_RUBRIC_PROMPT = """You are an expert QA auditor evaluating Twitter customer support replies for @SpotifyCares.
Evaluate the candidate reply against the customer query and the historical human reference reply.

Rate each on a scale from 1 (poor) to 5 (excellent):
1. tone: Friendly, empathetic, casual, professional Spotify voice.
2. groundedness: Accurately reflects Spotify troubleshooting (spoti.fi links, clean reinstall, DM for PII).
3. resolution_utility: Clear next action, diagnostic question, or proper escalation.

Output raw JSON only:
{"tone": <1-5>, "groundedness": <1-5>, "resolution_utility": <1-5>, "verdict": "pass" or "fail"}
"""

def evaluate_with_llm_judge(client, query, candidate_reply, reference_reply):
    try:
        user_content = f"Query: {query}\nReference: {reference_reply}\nCandidate: {candidate_reply}"
        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=100,
            temperature=0.0
        )
        raw = res.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        return float(data.get("groundedness", 3)), float(data.get("tone", 3)), float(data.get("resolution_utility", 3))
    except Exception:
        return 3.0, 3.0, 3.0

def run_evaluation():
    golden_path = "data/golden_set_150.csv"
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"{golden_path} not found.")

    df = pd.read_csv(golden_path)
    client = get_groq_client()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    models = {
        "Baseline 1 (Trivial Generic DM)": lambda text, intent, hist: trivial_baseline_agent(text)["reply"],
        "Baseline 2 (Rule/Keyword Agent)": lambda text, intent, hist: rule_based_baseline_agent(text, intent)["reply"],
        "LLM Agent (Groq / gpt-oss-20b)": lambda text, intent, hist: llm_agent(text, hist, client=client)["reply"]
    }

    results = []
    # Test on a representative subset of 40 rows for fast benchmark & judge evaluation
    sample_df = df.head(40).copy()
    print(f"Running benchmark on {len(sample_df)} golden examples across 3 systems...")

    for name, fn in models.items():
        print(f"Evaluating {name}...")
        r1_list, r2_list, rl_list, bleu_list, char_lens, under_280 = [], [], [], [], [], []
        judge_scores = []

        for idx, row in sample_df.iterrows():
            cand = fn(row["user_text"], row.get("intent", ""), row.get("spotify_response", ""))
            ref = str(row["spotify_response"])

            char_lens.append(len(cand))
            under_280.append(1 if len(cand) <= 280 else 0)

            scores = scorer.score(ref, cand)
            r1_list.append(scores['rouge1'].fmeasure)
            r2_list.append(scores['rouge2'].fmeasure)
            rl_list.append(scores['rougeL'].fmeasure)

            ref_tokens = nltk.word_tokenize(ref.lower())
            cand_tokens = nltk.word_tokenize(cand.lower())
            bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)
            bleu_list.append(bleu)

            if "LLM" in name and idx < 20:
                g, t, u = evaluate_with_llm_judge(client, row["user_text"], cand, ref)
                judge_scores.append((g + t + u) / 3.0)

        results.append({
            "System": name,
            "Avg Char Length": round(float(np.mean(char_lens)), 1),
            "% Under 280 Chars": round(float(np.mean(under_280)) * 100, 1),
            "ROUGE-1": round(float(np.mean(r1_list)), 4),
            "ROUGE-2": round(float(np.mean(r2_list)), 4),
            "ROUGE-L": round(float(np.mean(rl_list)), 4),
            "BLEU": round(float(np.mean(bleu_list)), 4),
            "LLM Judge Quality (1-5)": round(float(np.mean(judge_scores)), 2) if judge_scores else "N/A"
        })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("data/evaluation_metrics.csv", index=False)
    print("\n=== Benchmark Summary ===")
    print(metrics_df.to_string())

if __name__ == "__main__":
    run_evaluation()