import pandera.pandas as pa
from pandera.typing import Series

class TermsDataModel(pa.DataFrameModel):
    Id: Series[int] = pa.Field(nullable=False)
    Terms: Series[str] = pa.Field(nullable=False)
    NiceClass: Series[int] = pa.Field(nullable=False)


