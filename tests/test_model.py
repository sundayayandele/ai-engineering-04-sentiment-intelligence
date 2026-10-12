from sentiment_intelligence.model import build_pipeline


def test_fit_and_predict():
    model = build_pipeline()
    model.fit(["good experience", "wonderful experience", "bad experience", "awful experience"],
              ["positive", "positive", "negative", "negative"])
    result = model.predict_proba(["good experience"])[0]
    assert len(result) == 2
    assert abs(sum(result) - 1.0) < 1e-6
