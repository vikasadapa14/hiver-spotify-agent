import pandas as pd

def extract_spotify_pairs(raw_path="data/twcs.csv", output_path="data/spotify_pairs.csv"):
    print("Loading raw dataset...")
    df = pd.read_csv(raw_path)

    # Use Int64 nullable integer type so no '.0' float suffixes are added
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce").astype("Int64")

    print("Filtering Spotify inbound inquiries and agent responses...")
    # 1. Official responses sent by @SpotifyCares
    spotify_replies = df[df["author_id"].str.lower() == "spotifycares"].dropna(subset=["in_response_to_tweet_id"]).copy()

    # 2. Inbound user inquiries
    inbound_tweets = df[df["inbound"] == True].copy()

    # Pair the user tweet with Spotify's reply
    pairs = pd.merge(
        inbound_tweets,
        spotify_replies,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_user", "_spotify")
    )

    clean_pairs = pd.DataFrame({
        "user_tweet_id": pairs["tweet_id_user"],
        "user_text": pairs["text_user"],
        "spotify_tweet_id": pairs["tweet_id_spotify"],
        "spotify_response": pairs["text_spotify"],
        "created_at_user": pairs["created_at_user"],
        "created_at_spotify": pairs["created_at_spotify"]
    }).drop_duplicates(subset=["user_tweet_id"])

    clean_pairs.to_csv(output_path, index=False)
    print(f"Extracted {len(clean_pairs)} paired conversations -> {output_path}")

if __name__ == "__main__":
    extract_spotify_pairs()