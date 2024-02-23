import pandas as pd
from typing import Any, Optional
from pydantic import BaseModel, Field, validator


class TableData(BaseModel):
    columnName: str = Field(..., description="Column Name of the Table")
    sampleData: list[Any] = Field(..., description="Sample Data of the Table")


class SherlockTagsRequest(BaseModel):
    data: list[TableData] = Field(
        ..., min_items=1, description="Table Data - column name and sample data"
    )


class SherlockTagsResponse(BaseModel):
    tags: dict[str, list[str]] = Field(..., description="Tags Detected")
