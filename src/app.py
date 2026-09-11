import importlib.util
from pathlib import Path

# Dynamic import of 03_agent.py
agent_path = Path(__file__).resolve().parent / "03_agent.py"
spec = importlib.util.spec_from_file_location("agent_module", agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

baseline_keyword_agent = agent_module.baseline_keyword_agent
llm_agent = agent_module.llm_agent
get_groq_client = agent_module.get_groq_client

def main():
    print("=" * 60)
    print("SPOTIFY CARES SUPPORT AGENT - INTERACTIVE TEST CONSOLE")
    print("=" * 60)
    print("Type your simulated tweet issue below (or 'exit' to quit).\n")
    
    try:
        client = get_groq_client()
    except Exception as e:
        print(f"Warning: Groq client init failed ({e}). Running baseline only.")
        client = None

    while True:
        user_query = input("\nUser Tweet > ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Exiting console.")
            break

        print("\n" + "-" * 50)
        # 1. Baseline
        base_reply = baseline_keyword_agent(user_query, "General & Recommendations")
        print(f"[Baseline Agent]:\n{base_reply}\n(Len: {len(base_reply)} chars)")

        # 2. LLM Agent
        if client:
            try:
                llm_reply = llm_agent(user_query, client=client)
                print(f"\n[LLM Agent (@SpotifyCares)]:\n{llm_reply}\n(Len: {len(llm_reply)} chars)")
            except Exception as e:
                print(f"\n[LLM Agent Error]: {e}")
        print("-" * 50)

if __name__ == "__main__":
    main()