# MEMBER 1: load the Adult dataset DIRECTLY from the UCI site, clean it, and build the "text -> integer" converter.
# Source: https://archive.ics.uci.edu/dataset/2/adult  (dataset id 2). No other copy of the data is used.
# Paper Appendix C: each text column becomes integers. No one-hot, no scaling.
from functools import lru_cache
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


@lru_cache(maxsize=1)          # download once per run of the program, then reuse
def load_adult():
    """Return (X, y). X = clean feature table, y = 0/1 labels (1 = income over 50K)."""
    from ucimlrepo import fetch_ucirepo
    adult = fetch_ucirepo(id=2)                          # downloads from archive.ics.uci.edu
    df = adult.data.features.copy()
    df["income"] = adult.data.targets.iloc[:, 0]

    # remove extra spaces from text columns, and turn "?" into missing
    text_columns = [c for c in df.columns if not is_numeric_dtype(df[c])]
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": pd.NA, "nan": pd.NA})

    df = df.dropna().reset_index(drop=True)              # drop rows with missing values

    # labels: some rows end with a dot (">50K."), so remove it first
    y = df["income"].str.rstrip(".").eq(">50K").astype(int)
    X = df.drop(columns=["income", "fnlwgt"])            # fnlwgt is a census weight, not a person feature
    return X, y


def make_transformer(X_private):
    """Learn the text -> integer map from the PRIVATE pool only.
    A category the private pool never saw becomes -1 in the other splits."""
    text_cols = [c for c in X_private.columns if not is_numeric_dtype(X_private[c])]
    number_cols = [c for c in X_private.columns if is_numeric_dtype(X_private[c])]
    transformer = ColumnTransformer([
        ("text", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), text_cols),
        ("numbers", "passthrough", number_cols),         # numbers stay as they are (no scaling)
    ])
    transformer.fit(X_private)
    return transformer


if __name__ == "__main__":
    X, y = load_adult()
    print("rows:", len(X), " columns:", X.shape[1])
    print("share earning >50K:", round(y.mean(), 4))
    print(X.head(3))