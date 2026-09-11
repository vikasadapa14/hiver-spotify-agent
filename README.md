\# Spotify Customer Support Agent (@SpotifyCares)



An end-to-end NLP agent pipeline designed to automate customer support replies for Spotify on Twitter/X, benchmarked against real human support interactions from the Twitter Customer Support Dataset.



\---



\## 1. Project Architecture



hiver-spotify-agent/

│

├── data/

│   ├── twcs.csv                       # Raw source dataset

│   ├── spotify\_pairs.csv              # Extracted 41,585 Inbound-Reply pairs

│   ├── golden\_set\_150.csv             # 150-sample balanced benchmark set

│   ├── evaluation\_sample\_predictions.csv

│   └── evaluation\_metrics.csv         # Computed quantitative metrics

│

├── src/

│   ├── 01\_extract.py                  # Conversation reconstruction \& ID matching

│   ├── 02\_create\_golden\_set.py        # Intent categorization \& stratified sampling

│   ├── 03\_agent.py                    # Rule Baseline vs. LLM Agent implementation

│   ├── 04\_evaluate.py                 # ROUGE, BLEU, and constraint evaluation

│   └── app.py                         # Interactive testing CLI

│

├── requirements.txt

└── README.md





\---



\## 2. Methodology



\### A. Data Extraction \& Reconstruction

\- Filtered `@SpotifyCares` agent responses paired with initiating customer inquiries (`inbound=True`).

\- Handled floating-point tweet ID conversions using 64-bit integer casting to guarantee accurate relational joins.

\- Generated `41,585` validated customer-agent conversation pairs.



\### B. Golden Set Creation

\- Segmented user queries across 5 core support categories:

&#x20; 1. \*\*Account \& Login\*\*

&#x20; 2. \*\*Billing \& Subscription\*\*

&#x20; 3. \*\*Playback \& App Bugs\*\*

&#x20; 4. \*\*Offline \& Downloads\*\*

&#x20; 5. \*\*General Inquiries\*\*

\- Stratified sample of 30 balanced examples per intent to construct a standardized \*\*150-conversation evaluation benchmark\*\*.



\### C. Agents Implemented

1\. \*\*Rule-Based Baseline Agent\*\*: Deterministic keyword/intent routing providing immediate actionable links (e.g., `spoti.fi/reinstall`, account settings).

2\. \*\*LLM Support Agent\*\*: Prompt-engineered system mimicking `@SpotifyCares` persona, enforcing Twitter length restrictions ($\\le 280$ characters), friendly tone, and call-to-action (CTA) routing.



\---



\## 3. Quantitative Evaluation



Benchmark results comparing the Rule Baseline and LLM Agent against real human agent responses:



| Metric | Baseline (Keyword) | LLM Agent | Delta |

| :--- | :--- | :--- | :--- |

| \*\*ROUGE-1 (F1)\*\* | 0.2080 | \*\*0.2271\*\* | \*\*+9.2%\*\* |

| \*\*ROUGE-2 (F1)\*\* | 0.0283 | \*\*0.0590\*\* | \*\*+108.5%\*\* |

| \*\*ROUGE-L (F1)\*\* | 0.1480 | \*\*0.1728\*\* | \*\*+16.8%\*\* |

| \*\*BLEU Score\*\* | 0.0160 | \*\*0.0203\*\* | \*\*+26.9%\*\* |

| \*\*% Under 280 Chars\*\* | \*\*100.0%\*\* | 88.0% | -12.0% |

| \*\*% With Action/CTA\*\* | \*\*100.0%\*\* | 72.0% | -28.0% |



\### Key Takeaways \& Trade-offs

\- \*\*Contextual Adaptation\*\*: The LLM agent outperforms the baseline on every linguistic similarity metric (ROUGE-2 is over 2x higher), addressing specific device models and nuances (e.g., iPhone offload/cache directions) that keyword templates miss.

\- \*\*Safety \& Boundary Guarantees\*\*: While the LLM delivers higher conversational quality, the baseline guarantees 100% adherence to platform character limits and CTA triggers. 

\- \*\*Production Recommendation\*\*: A hybrid architecture—using the LLM for draft generation with a rule-based validation filter for character bounds and DM link injection.



\---



\## 4. Setup \& Execution



```powershell

\# 1. Activate Environment

.\\venv\\Scripts\\Activate.ps1



\# 2. Set API Key

$env:GROQ\_API\_KEY="your\_groq\_api\_key"



\# 3. Run Pipeline Steps

python src/01\_extract.py

python src/02\_create\_golden\_set.py

python src/03\_agent.py

python src/04\_evaluate.py



\# 4. Launch Interactive Console

python src/app.py



Save and close Notepad (\*\*Ctrl + S\*\*).



\---



\*\*Step 3: Run the Interactive Demo\*\*



Test the interactive console:



```powershell

python src\\app.py

