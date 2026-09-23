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


def test_citation_with_brackets_in_file_name():
    text = ('A good inference model shows the form of f ("DS-2 (1).pdf", p. 9). '
            'It finds important predictors ("DS-2 (1).pdf", p. 8, 9).')
    assert speakable(text) == "A good inference model shows the form of f. It finds important predictors."


def test_ordinary_brackets_are_kept():
    assert speakable("Linear models (like OLS) are fine for inference.") == \
        "Linear models (like OLS) are fine for inference."


def test_cite_parses_bracketed_and_quoted_names():
    from backend.citations import CITE
    text = ('f ("DS-2 (1).pdf", p. 9). g ("DS-2 (1).pdf", p. 8, 9). '
            'h (lecture4.pdf, p. 16). i (notes.pdf, pp. 12–17).')
    assert CITE.findall(text) == [("DS-2 (1).pdf", "9"), ("DS-2 (1).pdf", "8"),
                                  ("lecture4.pdf", "16"), ("notes.pdf", "12")]
