from typing import Optional, Union
import pandas as pd

from terms.schemas import TermsDataModel


def _clean(series_or_str: Union[pd.Series, str]) -> Union[pd.Series, str]:
    """
    Apply the canonical text-normalisation rules to either
    a pandas Series **or** a single raw string.
    """
    if isinstance(series_or_str, pd.Series):
        return series_or_str.str.strip().str.rstrip(".").str.lower().dropna()
    else:  
        return series_or_str.strip().rstrip(".").lower()


def preprocess(
    data: Union[str, pd.DataFrame],
    *,
    remove_duplicate_terms: bool = True,
    subsample_each_class_by: Optional[int] = None,
) -> Union[str, pd.DataFrame]:
    """
    Clean a trademark term or a whole trademark DataFrame.

    Parameters
    ----------
    data
        • `pd.DataFrame`  – processed row-wise (original behaviour)
        • `str`           – interpreted as **one** term and cleaned
    remove_duplicate_terms
        Only relevant when `data` is a DataFrame.
    subsample_each_class_by
        Ditto – ignored for single-term input.

    Returns
    -------
    str | pd.DataFrame
        • str input  ➜ cleaned **str** output
        • DataFrame ➜ cleaned **DataFrame**
    """
    # ─────────────────────────────────────────────
    # 1️⃣  DataFrame path  ─ same as your original
    # ─────────────────────────────────────────────
    if isinstance(data, pd.DataFrame):
        df = data.copy()

        # split & explode any semicolon-separated cell
        df[TermsDataModel.Terms] = df[TermsDataModel.Terms].str.split(";")
        df = df.explode(TermsDataModel.Terms)

        # text normalisation
        df[TermsDataModel.Terms] = _clean(df[TermsDataModel.Terms])

        df = df.dropna().reset_index(drop=True)

        if remove_duplicate_terms:
            df = df.drop_duplicates(subset=[TermsDataModel.Terms])

        if subsample_each_class_by:
            df = df.groupby(TermsDataModel.NiceClass, group_keys=False).head(
                subsample_each_class_by
            )

        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 2️⃣  Single-string path
    # ─────────────────────────────────────────────
    if isinstance(data, str):
        return _clean(data)

    # ─────────────────────────────────────────────
    # 3️⃣  Unsupported type
    # ─────────────────────────────────────────────
    raise TypeError(
        f"Unsupported type {type(data).__name__}. " "Expected str or pandas.DataFrame."
    )


def subsample(
    data_base: pd.DataFrame,
    data_complementary: pd.DataFrame,
    threshold_per_class: int = 400,
) -> pd.DataFrame:
    result = []

    classes = data_base[TermsDataModel.NiceClass].unique()

    for class_id in classes:
        base_terms = data_base[
            data_base[TermsDataModel.NiceClass] == class_id
        ].drop_duplicates(subset=TermsDataModel.Terms)
        selected = base_terms.head(threshold_per_class)

        if len(selected) < threshold_per_class:
            missing = threshold_per_class - len(selected)
            used_terms = set(selected[TermsDataModel.Terms])

            # Get complementary terms not already selected
            complementary_terms = (
                data_complementary[
                    (data_complementary[TermsDataModel.NiceClass] == class_id)
                    & (~data_complementary[TermsDataModel.Terms].isin(used_terms))
                ]
                .drop_duplicates(subset=TermsDataModel.Terms)
                .head(missing)
            )

            selected = pd.concat([selected, complementary_terms])

        result.append(selected)

    return pd.concat(result).reset_index(drop=True)
