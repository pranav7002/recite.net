from backend.speakable import speakable

DOC = '("DAI-101_Lecture_4 Variance_z score_preprocessing_missing data.pdf", p. 6)'


def test_citations_are_not_spoken():
    assert speakable(f"Divide by n minus one {DOC}.") == "Divide by n minus one."
    assert speakable("See the table (notes.pdf, pp. 12–17).") == "See the table."


def test_latex_is_read_as_words():
    out = speakable(r"Population variance ($\sigma^2$) divides by $n-1$, not $\bar{x}$.")
    assert out == "Population variance (sigma squared) divides by n minus 1, not x bar."
    assert "$" not in out and "\\" not in out


def test_unicode_maths_is_read_as_words():
    assert speakable("x̄ and σ² differ.") == "x bar and sigma squared differ."


def test_plain_text_is_unchanged():
    assert speakable("The median is robust to outliers.") == "The median is robust to outliers."
