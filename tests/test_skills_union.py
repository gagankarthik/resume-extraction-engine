"""
The union fields, which are arithmetic rather than judgement.

`technical_skills` and `all_skills_raw` used to be asked of the model, alongside
the buckets they summarise and the resume's own labels — so every skill was
written out three or four times. On a dense engineering resume that ran the
response past its ceiling, and a truncated response is re-asked from scratch
with double the budget: one agent spent minutes re-emitting the same words and
still lost the tail.

Deriving them here is faster and also strictly more correct. The model could
list a skill in a bucket and omit it from the union, and nothing downstream
would have noticed.
"""
from __future__ import annotations

from agents.skills import derive_union_fields


def test_technical_union_covers_every_technical_bucket():
    skills = derive_union_fields({
        "programming_languages": ["C#", "Python"],
        "frameworks_and_libraries": [".NET Core"],
        "databases": ["SQL Server"],
        "cloud_platforms": ["Azure"],
        "tools_and_platforms": ["Docker"],
        "operating_systems": ["Linux"],
        "methodologies": ["Agile"],
        "domain_skills": ["Machine Learning"],
        "design_skills": ["Figma"],
    })

    assert skills["technical_skills"] == [
        "C#", "Python", ".NET Core", "SQL Server", "Azure",
        "Docker", "Linux", "Agile", "Machine Learning", "Figma",
    ]


def test_soft_and_other_skills_are_in_all_but_not_technical():
    skills = derive_union_fields({
        "programming_languages": ["Python"],
        "soft_skills": ["Leadership"],
        "other_skills": ["Public Speaking"],
    })

    assert skills["technical_skills"] == ["Python"]
    assert skills["all_skills_raw"] == ["Python", "Leadership", "Public Speaking"]


def test_a_skill_in_two_buckets_appears_once():
    """The model is told not to, but the union must not depend on that."""
    skills = derive_union_fields({
        "programming_languages": ["SQL"],
        "databases": ["sql", "PostgreSQL"],
    })

    assert skills["technical_skills"] == ["SQL", "PostgreSQL"]


def test_first_spelling_wins():
    """Deduplication is case-insensitive; the resume's own casing is kept."""
    skills = derive_union_fields({
        "cloud_platforms": ["AWS"],
        "tools_and_platforms": ["aws", "Terraform"],
    })

    assert skills["technical_skills"] == ["AWS", "Terraform"]


def test_blank_and_non_string_entries_are_dropped():
    skills = derive_union_fields({
        "programming_languages": ["Python", "", "   ", None, 42],
    })

    assert skills["technical_skills"] == ["Python"]


def test_surrounding_whitespace_is_trimmed():
    skills = derive_union_fields({"databases": ["  Oracle  "]})

    assert skills["all_skills_raw"] == ["Oracle"]


def test_empty_input_yields_empty_unions():
    skills = derive_union_fields({})

    assert skills["technical_skills"] == []
    assert skills["all_skills_raw"] == []


def test_verbatim_categories_are_not_folded_into_the_unions():
    """categories[] is the resume's own labelling of the same skills.

    Folding it in would double-count nothing new, and would let a label like
    "ETL Tool" leak into the skills list as though it were a skill.
    """
    skills = derive_union_fields({
        "programming_languages": ["Python"],
        "categories": [{"name": "ETL Tool", "skills": ["Informatica"]}],
    })

    assert skills["all_skills_raw"] == ["Python"]
    assert skills["categories"] == [{"name": "ETL Tool", "skills": ["Informatica"]}]


def test_categories_carry_the_unions_when_the_taxonomy_pass_failed():
    """The two passes fail independently.

    When the taxonomy pass is the one that dies, every bucket is empty and the
    resume's own labelled categories are all that came back. Empty unions there
    read as "this candidate lists no skills" — the fallback is what keeps a
    real skills section from vanishing on a partial run.
    """
    skills = derive_union_fields({
        "programming_languages": [],
        "categories": [
            {"name": "Skill sets in SAP", "skills": ["Procure to Pay (PTP)", "QM"]},
            {"name": "Supporting tools", "skills": ["Solution Manager", "QM"]},
        ],
    })

    assert skills["technical_skills"] == ["Procure to Pay (PTP)", "QM", "Solution Manager"]
    assert skills["all_skills_raw"] == ["Procure to Pay (PTP)", "QM", "Solution Manager"]
