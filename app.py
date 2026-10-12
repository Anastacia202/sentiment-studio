"""
Sentiment Studio - Step 2: livelier single-review checker + batch dashboard
Needs only: streamlit, textblob  (pandas comes with streamlit)
Run:  python -m streamlit run sentiment_studio_v2.py
"""
import html
import re
from collections import Counter
from functools import lru_cache

import pandas as pd
import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="Sentiment Studio", page_icon="💬", layout="centered")

COLORS = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#F43F5E"}
EMOJI = {"Positive": "😍", "Neutral": "😐", "Negative": "😡"}
MESSAGE = {
    "Positive": "Customers are happy!",
    "Neutral": "Pretty balanced feedback.",
    "Negative": "This customer is unhappy.",
}
STOPWORDS = set(
    """a an the and or but if then so of to in on at for with from by is are was were be been it its this that these
    those i me my we our you your they them their he she his her as not no very just have has had do does did can
    could would should will than too also really got get one all about into out up there here what when which who am
    im ive dont didnt isnt wasnt product item order""".split()
)
EXAMPLES = {
    "😍 Love it": "I absolutely love this product! The quality is amazing and shipping was super fast.",
    "😡 Broken": "Terrible experience. It broke after two days and support never replied.",
    "😐 It's okay": "It's okay. Does the job, nothing special.",
    "🤔 Mixed": "Beautiful design but the delivery was very late. Disappointed with the wait.",
}
SAMPLE_REVIEWS = [
    "I absolutely love this product! The quality is amazing and shipping was super fast.",
    "Terrible experience. It broke after two days and support never replied.",
    "It's okay. Does the job, nothing special.",
    "Fantastic value for the money, would happily buy again!",
    "The packaging was damaged and the delivery was very late. Really disappointed.",
    "Great customer service, they fixed my issue within minutes.",
    "Average quality. Not bad, not great.",
    "Worst purchase ever. Cheap material and it looks nothing like the photos.",
    "Beautiful design and so comfortable to use. Highly recommend!",
    "Delivery took forever but the product itself is decent.",
    "Excellent! Exceeded my expectations in every way.",
    "Stopped working after a week. Waste of money.",
    "Easy to set up and works perfectly. Very happy with it.",
    "The instructions were confusing and the size was wrong.",
    "Pretty good overall, a few small issues but I like it.",
    "Awful smell and poor finish. I want a refund.",
    "Super friendly staff and fast shipping, five stars!",
    "Nothing remarkable. Arrived on time, works as described.",
    "Love the colours, it looks gorgeous in my living room.",
    "Disappointing quality for the price, very unhappy.",
    "Brilliant product, my whole family enjoys using it.",
    "Support was slow and unhelpful. Frustrating experience.",
    "Solid build and great battery life. Impressed!",
    "It's fine. I expected a bit more for this price.",
    "Perfect gift, the recipient was thrilled.",
    "Item arrived scratched and the replacement was also faulty.",
    "Smooth checkout, quick delivery, and the quality is wonderful.",
    "Mediocre at best. I wouldn't buy it again.",
    "Absolutely fantastic customer care and a lovely product.",
    "Horrible. Returned it the same day.",
]

# ---------- look and feel ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&display=swap');
.stApp {
    background: linear-gradient(135deg, #C7D2FE 0%, #E9D5FF 28%, #FBCFE8 58%, #FED7AA 100%);
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container {
    background: rgba(255, 255, 255, 0.74); border-radius: 30px;
    padding: 2rem 2.2rem 2.4rem 2.2rem !important; margin-top: 1.4rem; margin-bottom: 1.4rem;
    box-shadow: 0 24px 60px -26px #7C3AED99;
}
[data-testid="stWidgetLabel"] p { color: #312E81; font-weight: 600; }
.stTextArea textarea { background: white; border: 2px solid #C7D2FE; border-radius: 16px; color: #1E1B4B; }
.stTextArea textarea:focus { border-color: #7C3AED; }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }
.hero {
    background: linear-gradient(120deg, #4F46E5, #7C3AED 50%, #EC4899);
    border-radius: 26px; padding: 30px 34px; color: white; margin-bottom: 18px;
    box-shadow: 0 16px 34px -16px #7C3AED;
}
.hero h1 { color: white; margin: 0 0 6px 0; font-size: 2.4rem; }
.hero p { margin: 0; font-size: 1.05rem; opacity: .95; }
.stTabs [data-baseweb="tab"] {
    background: white; border-radius: 999px; padding: 8px 20px; height: auto;
    border: 2px solid #E0E7FF; font-weight: 700; color: #4F46E5;
}
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #4F46E5, #7C3AED); color: white !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stButton > button, .stDownloadButton > button {
    border-radius: 999px; font-weight: 700; border: 2px solid #C7D2FE; color: #4F46E5; background: white;
}
.stButton > button:hover { border-color: #7C3AED; color: #7C3AED; }
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4F46E5, #EC4899); color: white; border: none; padding: 8px 26px;
}
.result { border-radius: 22px; padding: 24px 28px; color: white; margin-top: 16px; }
.result .big { font-size: 3rem; line-height: 1; }
.result h2 { color: white; margin: 6px 0 2px 0; }
.meter { border-radius: 999px; height: 16px; margin: 20px 0 6px 0; position: relative; }
.meter .dot { position: absolute; top: -5px; width: 26px; height: 26px; border-radius: 50%;
              border: 4px solid white; box-shadow: 0 2px 8px #0003; }
.scale { display: flex; justify-content: space-between; color: #6B7280; font-size: .9rem; }
.card { background: white; border-radius: 20px; padding: 18px 22px; line-height: 2;
        box-shadow: 0 10px 28px -18px #4F46E5; }
.hl { padding: 2px 7px; border-radius: 8px; font-weight: 700; }
.hl.pos { background: #D1FAE5; color: #047857; }
.hl.neg { background: #FFE4E6; color: #BE123C; }
.kpi { background: white; border-radius: 20px; padding: 14px 16px; border-left: 8px solid var(--c);
       box-shadow: 0 10px 28px -18px #4F46E5; }
.kpi .v { font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 800; color: var(--c); }
.kpi .l { color: #6B7280; }
.happy { border-radius: 24px; padding: 22px 26px; color: white; text-align: center; margin: 14px 0; }
.happy .num { font-family: 'Outfit', sans-serif; font-size: 3.6rem; font-weight: 800; line-height: 1; }
.bar-row { display: flex; align-items: center; margin: 8px 0; gap: 10px; }
.bar-row .name { width: 90px; font-weight: 700; }
.bar-row .track { flex: 1; background: #EEF2FF; border-radius: 999px; height: 22px; overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 999px; }
.pill { display: inline-block; padding: 5px 13px; border-radius: 999px; margin: 3px 4px 3px 0; font-weight: 700; }
.pill.pos { background: #D1FAE5; color: #047857; }
.pill.neg { background: #FFE4E6; color: #BE123C; }
.quote { border-left: 6px solid var(--c); background: white; border-radius: 14px; padding: 10px 14px; margin: 8px 0; }
.insight { background: linear-gradient(135deg, #EEF2FF, #FDF2F8); border-radius: 20px; padding: 18px 24px; }
</style>
<div class="hero">
  <h1>💬 Sentiment Studio</h1>
  <p>See how your customers really feel, one review or thousands.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- analysis helpers ----------
def classify(polarity):
    if polarity >= 0.1:
        return "Positive"
    if polarity <= -0.1:
        return "Negative"
    return "Neutral"


def analyze(text):
    s = TextBlob(str(text)).sentiment
    return s.polarity, s.subjectivity, classify(s.polarity)


@lru_cache(maxsize=4096)
def word_polarity(word):
    return TextBlob(word).sentiment.polarity


def tokens(text):
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def highlight(text):
    out = []
    for part in re.split(r"([A-Za-z']+)", text):
        if re.fullmatch(r"[A-Za-z']+", part or ""):
            p = word_polarity(part.lower())
            if p >= 0.1:
                out.append('<span class="hl pos">' + html.escape(part) + "</span>")
                continue
            if p <= -0.1:
                out.append('<span class="hl neg">' + html.escape(part) + "</span>")
                continue
        out.append(html.escape(part))
    return "".join(out)


def distinctive_words(df, label, other, n=8):
    mine = Counter(w for t in df.loc[df["sentiment"] == label, "text"] for w in tokens(t))
    theirs = Counter(w for t in df.loc[df["sentiment"] == other, "text"] for w in tokens(t))
    scored = {w: c - theirs.get(w, 0) for w, c in mine.items()}
    ranked = sorted(scored.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, s in ranked if s > 0][:n]


def build_insights(df, pos_words, neg_words):
    total = len(df)
    share = df["sentiment"].value_counts(normalize=True).mul(100)
    share = share.reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    avg = df["polarity"].mean()
    lines = [
        "Out of %d reviews, %.0f%% are positive, %.0f%% neutral and %.0f%% negative."
        % (total, share["Positive"], share["Neutral"], share["Negative"]),
        "The average sentiment score is %+.2f (from -1 very negative to +1 very positive)." % avg,
    ]
    if avg >= 0.25:
        lines.append("Overall mood is strongly positive. Ask happy customers for public reviews.")
    elif avg >= 0.05:
        lines.append("Overall mood is mildly positive. Turn neutral customers into fans.")
    elif avg > -0.05:
        lines.append("Overall mood is mixed. Look at what splits happy and unhappy customers.")
    else:
        lines.append("Overall mood is negative. Fix the top complaints before new promotions.")
    if pos_words:
        lines.append("Customers praise: " + ", ".join(pos_words[:6]) + ".")
    if neg_words:
        lines.append("Customers complain about: " + ", ".join(neg_words[:6]) + ".")
    if "date" in df and df["date"].notna().sum() >= 6:
        d = df.dropna(subset=["date"]).sort_values("date")
        half = len(d) // 2
        first, second = d["polarity"].iloc[:half].mean(), d["polarity"].iloc[half:].mean()
        if abs(second - first) < 0.03:
            lines.append("Sentiment has stayed steady over time.")
        else:
            word = "improving" if second > first else "declining"
            lines.append("Sentiment is %s over time (%+.2f then %+.2f)." % (word, first, second))
    return lines


def report_markdown(df, lines):
    out = ["# Customer Sentiment Insights Report", "", "## Key findings", ""]
    out += ["- " + line for line in lines]
    out += ["", "## Method", "",
            "Each review is scored with TextBlob. Polarity runs from -1 (negative) to +1 (positive). "
            "+0.1 or higher is Positive, -0.1 or lower is Negative, anything between is Neutral.", "",
            "## Most positive reviews", ""]
    out += ["- " + t for t in df.nlargest(3, "polarity")["text"]]
    out += ["", "## Most negative reviews", ""]
    out += ["- " + t for t in df.nsmallest(3, "polarity")["text"]]
    out += ["", "## Limitations", "",
            "Word-based scoring can miss sarcasm and mixed feelings. Read the underlying reviews before acting."]
    return "\n".join(out)


# ---------- page ----------
tab_one, tab_many = st.tabs(["✍️ One review", "📊 Many reviews"])

with tab_one:
    st.session_state.setdefault("review", EXAMPLES["😍 Love it"])
    st.session_state.setdefault("auto", False)

    def pick(txt):
        st.session_state["review"] = txt
        st.session_state["auto"] = True

    st.text_area("Customer review", key="review", height=120)
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (name, txt) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=pick, args=(txt,), use_container_width=True)

    clicked = st.button("✨ Analyze sentiment", type="primary")

    if clicked or st.session_state["auto"]:
        st.session_state["auto"] = False
        text = st.session_state["review"].strip()
        if not text:
            st.warning("Type or paste a review first.")
        else:
            polarity, subjectivity, label = analyze(text)
            color = COLORS[label]
            position = (polarity + 1) / 2 * 100
            mood = "opinionated" if subjectivity >= 0.5 else "mostly factual"
            st.markdown(
                '<div class="result" style="background: linear-gradient(120deg, %s, %sCC);">'
                '<div class="big">%s</div><h2>%s: %s</h2>Score %+.2f &nbsp;|&nbsp; '
                "Subjectivity %.2f (%s)</div>"
                '<div class="meter" style="background: linear-gradient(90deg, #F43F5E, #F59E0B, #10B981);">'
                '<div class="dot" style="left: calc(%.0f%% - 13px); background: %s;"></div></div>'
                '<div class="scale"><span>Negative</span><span>Neutral</span><span>Positive</span></div>'
                % (color, color, EMOJI[label], label, MESSAGE[label], polarity, subjectivity, mood, position, color),
                unsafe_allow_html=True,
            )
            st.markdown("##### What drove the score")
            st.markdown('<div class="card">' + highlight(text) + "</div>", unsafe_allow_html=True)
            st.caption("Green words lift the score, red words pull it down.")
            if polarity >= 0.5:
                st.balloons()

with tab_many:
    st.markdown("#### Upload a CSV of reviews")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    use_sample = st.toggle("Use the sample dataset instead", value=up is None)

    data = None
    if up is not None and not use_sample:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error("We couldn't read that file (%s). Please upload a valid CSV." % e)
    elif use_sample:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(SAMPLE_REVIEWS), freq="D")
        data = pd.DataFrame({"date": dates, "review": SAMPLE_REVIEWS})

    if data is None or data.empty:
        st.info("Upload a CSV or switch on the sample dataset to see your dashboard.")
    else:
        text_cols = [c for c in data.columns if data[c].dtype == object]
        if not text_cols:
            st.error("No text column found. Your CSV needs a column with the review text.")
            st.stop()
        guess = next((c for c in text_cols if re.search(r"review|text|comment|feedback", c, re.I)), text_cols[0])
        date_guess = next((c for c in data.columns if re.search(r"date|time", c, re.I)), None)

        c1, c2 = st.columns(2)
        text_col = c1.selectbox("Review column", text_cols, index=text_cols.index(guess))
        options = ["(none)"] + list(data.columns)
        date_col = c2.selectbox("Date column (optional)", options,
                                index=options.index(date_guess) if date_guess else 0)

        df = pd.DataFrame({"text": data[text_col].astype(str)})
        if date_col != "(none)":
            df["date"] = pd.to_datetime(data[date_col], errors="coerce")
        df = df[df["text"].str.strip().ne("") & df["text"].ne("nan")].reset_index(drop=True)
        scores = df["text"].apply(analyze)
        df["polarity"] = [s[0] for s in scores]
        df["subjectivity"] = [s[1] for s in scores]
        df["sentiment"] = [s[2] for s in scores]

        total = len(df)
        counts = df["sentiment"].value_counts()
        happiness = round((df["polarity"].mean() + 1) / 2 * 100)
        tier = "Customers are delighted" if happiness >= 65 else "Customers are fairly content" if happiness >= 50 else "Customers need attention"
        face = "🤩" if happiness >= 65 else "🙂" if happiness >= 50 else "😟"
        hcolor = "#10B981" if happiness >= 65 else "#F59E0B" if happiness >= 50 else "#F43F5E"

        st.markdown(
            '<div class="happy" style="background: linear-gradient(120deg, %s, %sCC);">'
            '<div style="font-size:2.4rem">%s</div><div class="num">%d / 100</div>'
            "<div>Happiness score &middot; %s</div></div>" % (hcolor, hcolor, face, happiness, tier),
            unsafe_allow_html=True,
        )

        k = st.columns(4)
        kpis = [("Reviews", str(total), "#4F46E5")] + [
            (name, "%.0f%%" % (counts.get(name, 0) / total * 100), COLORS[name])
            for name in ["Positive", "Neutral", "Negative"]
        ]
        for col, (label, val, color) in zip(k, kpis):
            col.markdown('<div class="kpi" style="--c:%s"><div class="v">%s</div><div class="l">%s</div></div>'
                         % (color, val, label), unsafe_allow_html=True)

        st.markdown("##### How customers feel")
        rows = ""
        for name in ["Positive", "Neutral", "Negative"]:
            pct = counts.get(name, 0) / total * 100
            rows += ('<div class="bar-row"><div class="name">%s %s</div><div class="track">'
                     '<div class="fill" style="width:%.0f%%; background:%s"></div></div><div>%.0f%%</div></div>'
                     % (EMOJI[name], name, pct, COLORS[name], pct))
        st.markdown(rows, unsafe_allow_html=True)

        if "date" in df and df["date"].notna().sum() >= 3:
            st.markdown("##### Mood over time")
            daily = df.dropna(subset=["date"]).set_index("date").resample("D")["polarity"].mean().dropna()
            st.line_chart(daily, color="#EC4899", height=220)

        pos_words = distinctive_words(df, "Positive", "Negative")
        neg_words = distinctive_words(df, "Negative", "Positive")
        lines = build_insights(df, pos_words, neg_words)

        st.markdown("##### 💡 Insights at a glance")
        items = "".join("<li>" + html.escape(line) + "</li>" for line in lines)
        st.markdown('<div class="insight"><ul style="margin:0;padding-left:20px">' + items + "</ul></div>",
                    unsafe_allow_html=True)

        w1, w2 = st.columns(2)
        with w1:
            st.markdown("##### 💚 What they love")
            st.markdown("".join('<span class="pill pos">%s</span>' % html.escape(w) for w in pos_words) or "Not enough data.",
                        unsafe_allow_html=True)
        with w2:
            st.markdown("##### 💔 What frustrates them")
            st.markdown("".join('<span class="pill neg">%s</span>' % html.escape(w) for w in neg_words) or "Not enough data.",
                        unsafe_allow_html=True)

        st.markdown("##### 🗣️ In their own words")
        for t in df.nlargest(2, "polarity")["text"]:
            st.markdown('<div class="quote" style="--c:%s">%s</div>' % (COLORS["Positive"], html.escape(t)),
                        unsafe_allow_html=True)
        for t in df.nsmallest(2, "polarity")["text"]:
            st.markdown('<div class="quote" style="--c:%s">%s</div>' % (COLORS["Negative"], html.escape(t)),
                        unsafe_allow_html=True)

        with st.expander("See every scored review"):
            st.dataframe(df.round(2), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Scored CSV", df.to_csv(index=False).encode("utf-8"),
                           "scored_reviews.csv", "text/csv", use_container_width=True)
        d2.download_button("📝 Insights report", report_markdown(df, lines).encode("utf-8"),
                           "sentiment_insights_report.md", "text/markdown", use_container_width=True)
