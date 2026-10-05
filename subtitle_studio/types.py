from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator

TRANSLATION_VERSION = 2


class Settings(BaseModel):
    preset: Literal["accurate", "fast"] = "accurate"
    language: str | None = None
    recheck: bool = True
    debate: bool = True
    audio_profile: Literal["original", "level", "gentle", "speech"] = "level"
    recover_speech: bool = True
    chunk_seconds: int = Field(default=16, ge=8, le=60)
    batch_size: int = Field(default=4, ge=1, le=16)
    start_seconds: float = Field(default=0, ge=0)
    limit_seconds: float | None = Field(default=None, gt=0)
    target_language: str | None = None
    review_context: str | None = Field(default=None, max_length=4000)

    @field_validator("review_context")
    @classmethod
    def strip_context(cls, value):
        return value.strip() if value is not None else None


class ProjectContext(BaseModel):
    review_context: str = Field(max_length=4000)

    @field_validator("review_context")
    @classmethod
    def strip_context(cls, value):
        return value.strip()


class ProbeRequest(BaseModel):
    path: str


class JobRequest(BaseModel):
    tracks: list[int] = Field(min_length=1)
    settings: Settings = Field(default_factory=Settings)


class CueEdit(BaseModel):
    text: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    reviewed: bool | None = None


class ExportRequest(BaseModel):
    track: int
    language: str = "original"
    directory: str | None = None


class PlayRequest(BaseModel):
    track: int
    language: str = "original"
    start_seconds: float = Field(default=0, ge=0)
    audio_track: int | None = None


class SubtitleImportRequest(BaseModel):
    tracks: list[int] = Field(min_length=1)
    languages: dict[int, str] = Field(default_factory=dict)


class SubtitleSelection(BaseModel):
    track: int
    language: str = "original"
    default: bool = False


class RemuxRequest(BaseModel):
    subtitles: list[SubtitleSelection] = Field(min_length=1)
    output_path: str
    keep_embedded: bool = True
