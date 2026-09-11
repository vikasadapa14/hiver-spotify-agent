import re
import pandas as pd

def categorize_intent(text):
    text_lower = str(text).lower()
    
    # Keyword clusters for Spotify issues
    if re.search(r'\b(login|log in|password|reset|account|email|username|locked|2fa)\b', text_lower):
        return "Account & Login"
    elif re.search(r'\b(bill|charge|premium|subscription|student|family|refund|payment|cancel|card|receipt)\b', text_lower):
        return "Billing & Subscription"
    elif re.search(r'\b(crash|freeze|bug|error|skip|glitch|pause|sound|stuck|playing|stopped)\b', text_lower):
        return "Playback & Bugs"
    elif re.search(r'\b(download|offline|storage|sync|cache|save|data)\b', text_lower):
        return "Offline & Sync"
    else:
        return "General & Recommendations"

def create_golden_set(input_path="data/spotify_pairs.csv", output_path="data/golden_set_150.csv"):
    df = pd.read_csv(input_path)
    
    # Filter out empty or extremely brief text
    df = df[df["user_text"].str.split().str.len() >= 4].copy()
    
    # Assign intent
    df["intent"] = df["user_text"].apply(categorize_intent)
    
    print("Intent distribution across dataset:")
    print(df["intent"].value_counts())
    
    # Sample 30 items per intent category to make exactly 150 items
    golden_samples = []
    for intent, group in df.groupby("intent"):
        sample_count = min(30, len(group))
        golden_samples.append(group.sample(n=sample_count, random_state=42))
        
    golden_df = pd.concat(golden_samples).reset_index(drop=True)
    golden_df.to_csv(output_path, index=False)
    
    print(f"\nCreated golden benchmark set with {len(golden_df)} examples -> {output_path}")

if __name__ == "__main__":
    create_golden_set()