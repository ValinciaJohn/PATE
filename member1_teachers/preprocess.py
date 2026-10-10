# member1_teachers/preprocess.py
# MEMBER 1: load the Adult dataset from UCI, clean it, and build the feature transformer.
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CACHE_FILE = os.path.join("data", "raw", "adult.csv")


def load_adult():
    """Return (X, y). X = clean feature table, y = 0/1 labels (1 = income over 50K)."""
    if os.path.exists(CACHE_FILE):                       # already downloaded before
        df = pd.read_csv(CACHE_FILE)
    else:                                                # first time: download from UCI
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        df = adult.data.features.copy()
        df["income"] = adult.data.targets.iloc[:, 0]
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)               # save so we never download again

    # remove extra spaces from text columns, and turn "?" into missing
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": pd.NA, "nan": pd.NA})

    df = df.dropna().reset_index(drop=True)              # drop rows with missing values

    # labels: some rows end with a dot (">50K."), so remove it first
    y = df["income"].str.rstrip(".").eq(">50K").astype(int)
    X = df.drop(columns=["income", "fnlwgt"])            # fnlwgt is a census weight, not a person feature
    return X, y


def make_transformer(X_private):
    """Learn one-hot + scaling from the PRIVATE pool only, then reuse it on the other splits."""
    text_cols = X_private.select_dtypes(include="object").columns.tolist()
    number_cols = X_private.select_dtypes(exclude="object").columns.tolist()
    transformer = ColumnTransformer([
        ("text", OneHotEncoder(handle_unknown="ignore", sparse_output=False), text_cols),
        ("numbers", StandardScaler(), number_cols),
    ])
    transformer.fit(X_private)
    return transformer


if __name__ == "__main__":
    X, y = load_adult()
    print("rows:", len(X), " columns:", X.shape[1])
    print("share earning >50K:", round(y.mean(), 4))
    print(X.head(3))