import streamlit as st
from transformers import pipeline
from pytrends.request import TrendReq
import pandas as pd

# Load text generation model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Hook/Idea Generator", page_icon="🎬")
st.title("🎬 AI Hook/Idea Generator for Reels & TikToks")
st.write("Generate engaging video ideas tailored for your niche — now trend-aware!")

# Inputs
niche = st.text_input("Enter a niche (e.g., 'Math tricks', 'AI tools', 'Language learning')", "")
tone = st.selectbox("Select tone", ["Educational", "Funny", "Motivational", "Trendy", "Informative"])
num_ideas = st.slider("Number of ideas to generate", 3, 10, 5)
trend_mode = st.checkbox("Enable Trend-Aware Mode (uses Google Trends)")

# Fetch trends
def fetch_trends(keyword):
    pytrends = TrendReq(hl="en-US", tz=330)
    pytrends.build_payload([keyword], cat=0, timeframe="today 3-m", geo="US", gprop="")
    df = pytrends.interest_over_time()
    if not df.empty:
        avg_interest = df[keyword].mean()
        st.write(f"📈 Average search interest for '{keyword}' (past 3 months): **{avg_interest:.2f}**")
        return avg_interest
    else:
        st.warning("No trend data found. Continuing without trend adjustment.")
        return None

if st.button("Generate Ideas"):
    if not niche:
        st.warning("Please enter a niche.")
    else:
        trend_text = ""
        if trend_mode:
            st.info("Fetching real-time trends from Google...")
            interest = fetch_trends(niche)
            if interest and interest > 30:
                trend_text = "This topic is trending strongly, focus on viral and attention-grabbing ideas."
            elif interest and interest < 10:
                trend_text = "This topic is less active, generate evergreen and informative ideas."
            else:
                trend_text = "Generate moderately trending, creative ideas."
        
        prompt = f"Generate {num_ideas} short, catchy TikTok or Reels video ideas about {niche} in a {tone.lower()} tone. {trend_text} Each idea should be unique and engaging."

        with st.spinner("Generating ideas..."):
            result = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.9, do_sample=True)
            text = result[0]["generated_text"]

        ideas = text.split("\n")
        st.subheader("✨ Generated Ideas")
        for i, idea in enumerate(ideas[:num_ideas], start=1):
            st.markdown(f"**{i}.** {idea.strip()}")

st.caption("Powered by Hugging Face GPT-2 and Google Trends (pytrends)")
