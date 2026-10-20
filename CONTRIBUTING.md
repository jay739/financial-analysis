# Contributing to financial-analysis

Thanks for considering a contribution. Issues, discussions and PRs are all
welcome.

## Getting set up

Python 3.10 or newer. `pip install -r requirements.txt`, then `streamlit run app.py`. FinBERT (torch and transformers, about 400 MB) only downloads if you tick the FinBERT box in the sidebar.

## Running the checks

`pip install -r requirements-test.txt` then `pytest -q tests`. The suite stubs yfinance, so it needs no network. CI runs it on every PR.

## Ground rules

- Open an issue or discussion before large changes, so effort isn't wasted.
- Keep PRs focused: one change per PR.
- New behavior needs a test where the repo has a test suite.
- CI must pass before merge.

## Licensing

By contributing you agree your contributions are licensed under the repo's
MIT license.
