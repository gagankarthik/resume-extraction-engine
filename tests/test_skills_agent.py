"""
The two-step skills pipeline: read the inventory, then sort it.

The step that mattered is the second one's INPUT. It used to read the whole
resume and emit every technology named anywhere in it — four hundred entries
from a thirty-year consulting resume, measured at 85-250 seconds against a
150-second budget for the entire extraction. It timed out on most runs, and a
resume with a full skills section arrived with thirteen empty buckets. It now
sorts the list step one already found, so its cost is bounded by that list.

Driven against a fake provider, so there is no network and no API key.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agents.llm.client import LLMClient
from agents.llm.providers import Completion
from agents.skills import SkillsAgent
from config import LLMSettings

LABELLED_RESUME = """
Jane Doe

TECHNICAL SKILLS
ETL Tool: Informatica, DataStage
Cloud Datawarehouse: Snowflake, Redshift

PROFESSIONAL EXPERIENCE
Acme Corporation
• Ran the nightly load using a scheduler and a reporting stack
"""


class FakeProvider:
    """Answers each prompt in the shape its agent expects, and records them."""

    name = "fake"

    def __init__(self, *, inventory: dict, taxonomy: dict, fail: set[str] | None = None):
        self.inventory = inventory
        self.taxonomy = taxonomy
        self.fail = fail or set()
        self.seen: list[tuple[str, str]] = []

    async def complete(self, system, user, *, max_tokens, temperature, json_mode, timeout=None):
        step = self._step(system)
        self.seen.append((step, user))
        if step in self.fail:
            raise RuntimeError(f"{step} unavailable")
        payload = {"inventory": self.inventory, "taxonomy": self.taxonomy,
                   "document": self.taxonomy}[step]
        return Completion(
            text=json.dumps(payload), input_tokens=10, output_tokens=10,
            truncated=False, model="fake", provider=self.name,
        )

    @staticmethod
    def _step(system: str) -> str:
        if "given the list of skills read off" in system:
            return "taxonomy"
        if "Preserve the resume's OWN skills-section labels" in system:
            return "inventory"
        return "document"


def _settings() -> LLMSettings:
    return LLMSettings(
        model="fake",
        max_concurrent=4, max_output_tokens=32000, call_timeout_seconds=30,
        transport_retries=0, truncation_escalations=1,
    )


@pytest.fixture
def run_skills(monkeypatch):
    def go(provider: FakeProvider, text: str = LABELLED_RESUME) -> dict:
        client = LLMClient(settings=_settings(), provider=provider)
        monkeypatch.setattr("agents.llm.client.get_client", lambda: client)
        monkeypatch.setattr("agents.base.get_client", lambda: client)
        return asyncio.run(SkillsAgent().run(text))

    return go


def test_the_sorting_step_never_sees_the_resume(run_skills):
    """The whole point of the rework.

    Handing the taxonomy step the document is what made it generate skills out
    of job prose, and what made it slow enough to time out.
    """
    provider = FakeProvider(
        inventory={"categories": [{"name": "ETL Tool", "skills": ["Informatica", "DataStage"]}],
                   "uncategorized": []},
        taxonomy={"skills": {"tools_and_platforms": ["Informatica", "DataStage"]}},
    )
    run_skills(provider)

    steps = [step for step, _ in provider.seen]
    assert steps == ["inventory", "taxonomy"], steps

    taxonomy_input = next(user for step, user in provider.seen if step == "taxonomy")
    assert "Informatica" in taxonomy_input
    assert "nightly load" not in taxonomy_input, "the sorting step was handed the resume"
    assert "PROFESSIONAL EXPERIENCE" not in taxonomy_input


def test_labels_and_buckets_describe_the_same_skills(run_skills):
    provider = FakeProvider(
        inventory={"categories": [
            {"name": "ETL Tool", "skills": ["Informatica"]},
            {"name": "Cloud Datawarehouse", "skills": ["Snowflake"]},
        ], "uncategorized": []},
        taxonomy={"skills": {"tools_and_platforms": ["Informatica"],
                             "cloud_platforms": ["Snowflake"]}},
    )
    skills = run_skills(provider)

    assert [c["name"] for c in skills["categories"]] == ["ETL Tool", "Cloud Datawarehouse"]
    assert skills["technical_skills"] == ["Snowflake", "Informatica"]
    assert skills["all_skills_raw"] == ["Snowflake", "Informatica"]


def test_an_unlabelled_skills_list_still_gets_sorted(run_skills):
    provider = FakeProvider(
        inventory={"categories": [], "uncategorized": ["Python", "Oracle"]},
        taxonomy={"skills": {"programming_languages": ["Python"], "databases": ["Oracle"]}},
    )
    skills = run_skills(provider)

    assert [step for step, _ in provider.seen] == ["inventory", "taxonomy"]
    assert skills["programming_languages"] == ["Python"]
    # No labels in the resume means no categories[] — the validator defaults it.
    assert skills.get("categories", []) == []


def test_a_resume_with_no_skills_section_falls_back_to_the_document(run_skills):
    """Nothing to sort means the old behaviour is the only behaviour left."""
    provider = FakeProvider(
        inventory={"categories": [], "uncategorized": []},
        taxonomy={"skills": {"tools_and_platforms": ["Jenkins"]}},
    )
    skills = run_skills(provider)

    assert [step for step, _ in provider.seen] == ["inventory", "document"]
    assert skills["tools_and_platforms"] == ["Jenkins"]


def test_a_failed_sort_keeps_the_skills_ungrouped(run_skills):
    """Grouping is the only thing a failed second step may cost."""
    provider = FakeProvider(
        inventory={"categories": [{"name": "ETL Tool", "skills": ["Informatica"]}],
                   "uncategorized": ["Python"]},
        taxonomy={},
        fail={"taxonomy"},
    )
    skills = run_skills(provider)

    assert skills["all_skills_raw"] == ["Informatica", "Python"]
    assert skills["categories"] == [{"name": "ETL Tool", "skills": ["Informatica"]}]


def test_a_failed_inventory_still_reads_the_document(run_skills):
    provider = FakeProvider(
        inventory={},
        taxonomy={"skills": {"databases": ["Oracle"]}},
        fail={"inventory"},
    )
    skills = run_skills(provider)

    assert [step for step, _ in provider.seen] == ["inventory", "document"]
    assert skills["databases"] == ["Oracle"]


def test_packed_and_prose_entries_are_cleaned_on_the_way_in(run_skills):
    provider = FakeProvider(
        inventory={"categories": [{"name": "Stack", "skills": [
            "App Servers: WebSphere, WebLogic",
            "Plan to produce (PTM PP, PP-PI)",
            "Work Authorization - US Permanent Resident (Green Card). No sponsorship required.",
        ]}], "uncategorized": []},
        taxonomy={"skills": {"other_skills": ["WebSphere", "WebLogic",
                                              "Plan to produce (PTM PP, PP-PI)"]}},
    )
    skills = run_skills(provider)

    assert skills["categories"][0]["skills"] == [
        "WebSphere", "WebLogic", "Plan to produce (PTM PP, PP-PI)",
    ]
    taxonomy_input = next(user for step, user in provider.seen if step == "taxonomy")
    assert "sponsorship" not in taxonomy_input


def test_a_jobs_environment_line_is_not_a_skills_category(run_skills):
    """The tech stack under one job belongs to that job, not to Technical Skills.

    The inventory pass reads the whole document — it has to, because real
    resumes put skills blocks between the work sections — so "Environment:
    Tosca, SQL Developer" under a job arrives looking exactly like a labelled
    skills category. It is already extracted as that job's technologies and
    printed beneath it, and one submitted resume came out with four Technical
    Skills categories all called "Environment", one per job.
    """
    provider = FakeProvider(
        inventory={
            "categories": [
                {"name": "TOSCA Platform", "skills": ["Tricentis TOSCA AS1"]},
                {"name": "Environment", "skills": ["Tosca", "SQL Developer"]},
                {"name": "Key Technologies/Skills", "skills": ["QTP", "Sybase"]},
            ],
            "uncategorized": [],
        },
        taxonomy={"skills": {"other_skills": ["Tricentis TOSCA AS1"]}},
    )
    skills = run_skills(provider)

    assert [c["name"] for c in skills["categories"]] == ["TOSCA Platform"]


def test_a_real_category_that_merely_starts_with_environment_is_kept(run_skills):
    """"Environment Management" is a discipline, not a job's tech line."""
    provider = FakeProvider(
        inventory={
            "categories": [{"name": "Environment Management", "skills": ["Terraform"]}],
            "uncategorized": [],
        },
        taxonomy={"skills": {"other_skills": ["Terraform"]}},
    )
    skills = run_skills(provider)

    assert [c["name"] for c in skills["categories"]] == ["Environment Management"]
