"""
Pydantic v2 validation layer for LLM-extracted resume JSON.
Every section and every list item has a typed model.
Rules:
  - extra="allow"  → never discard fields the LLM added outside the schema
  - coerce helpers  → handle type mismatches gracefully (str↔list, null↔"")
  - on failure      → return original data rather than losing it
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Primitive coercion helpers
# ---------------------------------------------------------------------------

def _str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(v)]


def _float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v)) if v is not None else None
    except (ValueError, TypeError):
        return None


def _bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return bool(v)


def _str_list(v: Any) -> list[str]:
    return [str(i).strip() for i in _list(v) if i and str(i).strip()]


# ---------------------------------------------------------------------------
# Base class — shared config
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = {"extra": "allow", "populate_by_name": True}


# ---------------------------------------------------------------------------
# personal_information
# ---------------------------------------------------------------------------

class AddressModel(_Base):
    full_address: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


class PersonalInformationModel(_Base):
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: list[str] = Field(default_factory=list)
    phone: list[str] = Field(default_factory=list)
    address: Optional[AddressModel] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    twitter_url: Optional[str] = None
    other_urls: list[str] = Field(default_factory=list)
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    profile_headline: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("email", "phone", "other_urls"):
                out[k] = _str_list(v)
            elif k == "address":
                out[k] = v if isinstance(v, dict) else None
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# work_experience
# ---------------------------------------------------------------------------

class WorkExperienceItem(_Base):
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    employment_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None
    duration: Optional[str] = None
    location: Optional[str] = None
    remote: Optional[bool] = None
    department: Optional[str] = None
    reporting_to: Optional[str] = None
    team_size: Optional[int] = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies_used: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("responsibilities", "achievements", "technologies_used"):
                out[k] = _str_list(v)
            elif k in ("is_current", "remote"):
                out[k] = _bool(v)
            elif k == "team_size":
                out[k] = _int(v)
            elif isinstance(v, (list, dict)):
                # Preserve nested structures (projects[], subsections[], etc.)
                # CRITICAL: do NOT stringify — that's what was producing
                # "projects": "[{'projectName': ...}]" Python-repr blobs.
                out[k] = v
            else:
                out[k] = _str(v)

        # If the LLM mis-detected "sub-projects" and stuffed all duty bullets into
        # projects[].projectResponsibilities while leaving responsibilities[] empty,
        # flatten them back. Real per-project bullets are flat duty lines, not
        # discrete consulting engagements.
        projects = out.get("projects")
        resps = out.get("responsibilities") or []
        if not resps and isinstance(projects, list) and projects:
            # Heuristic: if every "project" has at most ONE projectResponsibility,
            # this isn't a real consulting structure — it's just bullets that the
            # LLM split into pseudo-projects. Flatten back.
            singleton_count = sum(
                1 for p in projects
                if isinstance(p, dict) and len(p.get("projectResponsibilities") or []) <= 1
            )
            if projects and singleton_count >= max(1, len(projects) // 2):
                flat: list[str] = []
                for p in projects:
                    if isinstance(p, dict):
                        flat.extend(p.get("projectResponsibilities") or [])
                if flat:
                    out["responsibilities"] = [str(x).strip() for x in flat if x and str(x).strip()]
                    out["projects"] = []

        return out


# ---------------------------------------------------------------------------
# education
# ---------------------------------------------------------------------------

class EducationItem(_Base):
    institution_name: Optional[str] = None
    degree: Optional[str] = None
    degree_type: Optional[str] = None
    field_of_study: Optional[str] = None
    major: Optional[str] = None
    minor: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None
    gpa: Optional[float] = None
    percentage: Optional[float] = None
    grade: Optional[str] = None
    honors: list[str] = Field(default_factory=list)
    relevant_coursework: list[str] = Field(default_factory=list)
    thesis_title: Optional[str] = None
    dissertation: Optional[str] = None
    location: Optional[str] = None
    activities: list[str] = Field(default_factory=list)
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("honors", "relevant_coursework", "activities"):
                out[k] = _str_list(v)
            elif k == "is_current":
                out[k] = _bool(v)
            elif k in ("gpa", "percentage"):
                out[k] = _float(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

class SkillCategoryItem(_Base):
    """Verbatim skill section from the resume — preserves the candidate's own label."""
    name: Optional[str] = None
    skills: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {
            "name": _str(d.get("name") or d.get("label") or d.get("category")),
            "skills": _str_list(d.get("skills") or d.get("items") or d.get("values")),
        }


class SkillsModel(_Base):
    all_skills_raw: list[str] = Field(default_factory=list)
    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks_and_libraries: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    cloud_platforms: list[str] = Field(default_factory=list)
    tools_and_platforms: list[str] = Field(default_factory=list)
    operating_systems: list[str] = Field(default_factory=list)
    methodologies: list[str] = Field(default_factory=list)
    domain_skills: list[str] = Field(default_factory=list)
    design_skills: list[str] = Field(default_factory=list)
    languages_spoken: list[str] = Field(default_factory=list)
    other_skills: list[str] = Field(default_factory=list)
    # Verbatim categories preserve the resume's own section labels.
    # Frontend prefers these over the normalized fields above when populated.
    categories: list[SkillCategoryItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "categories":
                # Pass list-of-dicts through untouched; SkillCategoryItem will coerce.
                out[k] = v if isinstance(v, list) else []
            else:
                out[k] = _str_list(v)
        return out


# ---------------------------------------------------------------------------
# certifications
# ---------------------------------------------------------------------------

class CertificationItem(_Base):
    name: Optional[str] = None
    issuing_organization: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    credential_url: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------

class ProjectItem(_Base):
    name: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None
    technologies: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    repository_url: Optional[str] = None
    highlights: list[str] = Field(default_factory=list)
    team_size: Optional[int] = None
    type: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("technologies", "highlights"):
                out[k] = _str_list(v)
            elif k == "is_current":
                out[k] = _bool(v)
            elif k == "team_size":
                out[k] = _int(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# publications
# ---------------------------------------------------------------------------

class PublicationItem(_Base):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "authors":
                out[k] = _str_list(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# awards_and_honors
# ---------------------------------------------------------------------------

class AwardItem(_Base):
    title: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    level: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# volunteer_experience
# ---------------------------------------------------------------------------

class VolunteerItem(_Base):
    organization: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    cause: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "responsibilities":
                out[k] = _str_list(v)
            elif k == "is_current":
                out[k] = _bool(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------

class LanguageItem(_Base):
    language: Optional[str] = None
    proficiency: Optional[str] = None
    reading: Optional[str] = None
    writing: Optional[str] = None
    speaking: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

class ReferenceItem(_Base):
    name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    relationship: Optional[str] = None
    available_on_request: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "available_on_request":
                out[k] = _bool(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# patents
# ---------------------------------------------------------------------------

class PatentItem(_Base):
    title: Optional[str] = None
    patent_number: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    inventors: list[str] = Field(default_factory=list)
    url: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "inventors":
                out[k] = _str_list(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# professional_memberships
# ---------------------------------------------------------------------------

class MembershipItem(_Base):
    organization: Optional[str] = None
    role: Optional[str] = None
    membership_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "is_current":
                out[k] = _bool(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# conferences_and_talks
# ---------------------------------------------------------------------------

class ConferenceItem(_Base):
    title: Optional[str] = None
    event: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# courses
# ---------------------------------------------------------------------------

class CourseItem(_Base):
    name: Optional[str] = None
    provider: Optional[str] = None
    platform: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    credential_id: Optional[str] = None
    duration: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

class TrainingItem(_Base):
    name: Optional[str] = None
    provider: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# extracurricular_activities
# ---------------------------------------------------------------------------

class ExtracurricularItem(_Base):
    organization: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {k: _str(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

class AnalyticsModel(_Base):
    total_years_of_experience: Optional[float] = None
    total_months_of_experience: Optional[int] = None
    career_level: Optional[str] = None
    primary_industry: Optional[str] = None
    secondary_industries: list[str] = Field(default_factory=list)
    job_functions: list[str] = Field(default_factory=list)
    highest_education_level: Optional[str] = None
    number_of_companies: Optional[int] = None
    number_of_roles: Optional[int] = None
    average_tenure_months: Optional[int] = None
    has_international_experience: Optional[bool] = None
    primary_location: Optional[str] = None
    salary_mentioned: Optional[str] = None
    resume_language: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k in ("secondary_industries", "job_functions"):
                out[k] = _str_list(v)
            elif k == "total_years_of_experience":
                out[k] = _float(v)
            elif k in ("total_months_of_experience", "number_of_companies",
                       "number_of_roles", "average_tenure_months"):
                out[k] = _int(v)
            elif k == "has_international_experience":
                out[k] = _bool(v)
            else:
                out[k] = _str(v)
        return out


# ---------------------------------------------------------------------------
# raw_sections
# ---------------------------------------------------------------------------

class RawSectionsModel(_Base):
    section_names_found: list[str] = Field(default_factory=list)
    unclassified_content: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        return {
            "section_names_found": _str_list(d.get("section_names_found")),
            "unclassified_content": _str(d.get("unclassified_content")),
        }


# ---------------------------------------------------------------------------
# Top-level resume model
# ---------------------------------------------------------------------------

def _item_list(model_cls, v: Any) -> list:
    """Coerce a raw value into a validated list of model_cls instances."""
    items = _list(v)
    out = []
    for item in items:
        if isinstance(item, dict):
            try:
                out.append(model_cls.model_validate(item))
            except Exception:
                out.append(model_cls.model_validate({}))
        else:
            out.append(model_cls.model_validate({"value": str(item)}))
    return out


class ResumeSchema(_Base):
    personal_information: Optional[PersonalInformationModel] = None
    professional_summary: Optional[str] = None
    objective: Optional[str] = None
    work_experience: list[WorkExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    skills: Optional[SkillsModel] = None
    certifications: list[CertificationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    publications: list[PublicationItem] = Field(default_factory=list)
    awards_and_honors: list[AwardItem] = Field(default_factory=list)
    volunteer_experience: list[VolunteerItem] = Field(default_factory=list)
    languages: list[LanguageItem] = Field(default_factory=list)
    interests_and_hobbies: list[str] = Field(default_factory=list)
    references: list[ReferenceItem] = Field(default_factory=list)
    patents: list[PatentItem] = Field(default_factory=list)
    professional_memberships: list[MembershipItem] = Field(default_factory=list)
    conferences_and_talks: list[ConferenceItem] = Field(default_factory=list)
    courses: list[CourseItem] = Field(default_factory=list)
    training: list[TrainingItem] = Field(default_factory=list)
    extracurricular_activities: list[ExtracurricularItem] = Field(default_factory=list)
    analytics: Optional[AnalyticsModel] = None
    raw_sections: Optional[RawSectionsModel] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, d: Any) -> Any:
        if not isinstance(d, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "professional_summary" or k == "objective":
                out[k] = _str(v)
            elif k == "interests_and_hobbies":
                out[k] = _str_list(v)
            else:
                out[k] = v  # typed list fields and sub-models handled by Pydantic
        return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_resume_json(raw: dict) -> tuple[dict, list[str]]:
    """
    Validate and clean LLM-extracted resume JSON.
    Returns (cleaned_dict, warnings).
    Never discards data — falls back to original on parse failure.
    """
    warnings: list[str] = []
    metadata = raw.pop("_metadata", None)

    try:
        model = ResumeSchema.model_validate(raw)
        cleaned = model.model_dump(mode="json", exclude_none=False)
    except Exception as exc:
        warnings.append(f"Top-level validation failed: {exc}")
        cleaned = raw

    if metadata is not None:
        cleaned["_metadata"] = metadata
        if warnings:
            cleaned["_metadata"]["validation_warnings"] = warnings

    return cleaned, warnings
