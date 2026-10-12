# Sentiment Studio

A colourful customer sentiment analysis tool built with Python and Streamlit. Paste a single review or upload a CSV of reviews and get instant sentiment scores, a dashboard, and plain-English insights.

**Live app:** _add your Streamlit link here after deploying_

![Screenshot](screenshot.png)

## Features

- **One review:** paste text and see the sentiment label, score, a negative-to-positive meter, and the words that raised or lowered the score.
- **Many reviews:** upload a CSV (or use the built-in sample) to get a happiness score, sentiment breakdown, mood over time, top praised and criticised words, and example quotes.
- **Downloads:** export the scored reviews as CSV and a ready-made insights report (Markdown).

## How it works (technical explanation)

1. Each review is scored with [TextBlob](https://textblob.readthedocs.io/), a lexicon-based library.
   - **Polarity** runs from -1 (very negative) to +1 (very positive).
   - **Subjectivity** runs from 0 (factual) to 1 (opinionated).
2. Polarity is turned into a label: **Positive** at +0.1 or higher, **Negative** at -0.1 or lower, otherwise **Neutral**.
3. For a batch, the app computes:
   - the share of each label and the average polarity,
   - a **happiness score** out of 100, calculated as `(average polarity + 1) / 2 * 100`,
   - the words that appear far more in positive reviews than negative ones (and the reverse),
   - a mood trend by comparing the first and second half of the reviews when a date column exists.
4. Streamlit renders the interface, with custom CSS for the colours and layout.

## Run it locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## CSV format

One column with the review text (for example `review`). An optional `date` column adds the trend chart.

## Limitations

Lexicon-based scoring can miss sarcasm, slang and mixed feelings in a single review, so results are a guide rather than a verdict. A future version could use a transformer model for higher accuracy.

## Tech stack

Python, Streamlit, TextBlob, pandas.
