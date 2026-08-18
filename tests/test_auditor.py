"""
Tests for the deterministic parts of CompletenessAuditorAgent.
No API keys or network needed. Run directly:  python tests/test_auditor.py
(also compatible with pytest if installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.auditor import (
    _CERTIFICATION_HEADING,
    _EXPERIENCE_HEADING,
    _figure_key,
    _figures,
    _squash,
    _tokens,
    coverage_report,
    ground_check,
    has_section,
    is_grounded,
    merge_recovered,
    merge_split_bullets,
    repair_figures,
    scrub_skills,
    source_bullet_blocks,
    source_line_index,
    source_terms,
    strip_dropped_sections,
    tech_is_grounded,
)

RESUME_TEXT = """
John Smith
john.smith@example.com | (555) 123-4567 | linkedin.com/in/john-smith

WORK EXPERIENCE
Acme Corporation                                Jan 2020 - Present
Senior Data Engineer
Client: Diligent Insurance
• Designed and implemented scalable ETL pipelines using Apache Spark and Airflow
• Reduced nightly batch processing time by 40 percent across all regions
• Mentored four junior engineers on data modeling best practices

EDUCATION
Bachelor of Science in Computer Science, State University, 2015
"""


def test_grounded_contacts_kept_and_fakes_dropped():
    merged = {
        "personal_information": {
            "email": ["john.smith@example.com", "fake.person@nowhere.org"],
            "phone": ["(555) 123-4567", "999-888-7777"],
            "linkedin_url": "https://linkedin.com/in/john-smith",
            "github_url": "https://github.com/totally-invented-handle",
        },
        "work_experience": [],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    pi = merged["personal_information"]
    assert pi["email"] == ["john.smith@example.com"], pi["email"]
    assert pi["phone"] == ["(555) 123-4567"], pi["phone"]
    assert pi["linkedin_url"] is not None
    assert pi["github_url"] is None
    assert len(warnings) == 3, warnings


def test_ungrounded_client_name_removed():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": ["Designed and implemented scalable ETL pipelines"],
            "projects": [{
                "projectName": "Diligent Insurance",       # grounded — appears in text
                "clientName": "Deloitte",                  # hallucinated — NOT in text
                "projectResponsibilities": ["Reduced nightly batch processing time by 40 percent"],
            }],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    proj = merged["work_experience"][0]["projects"][0]
    assert proj["clientName"] is None
    assert proj["projectName"] == "Diligent Insurance"
    assert any("Deloitte" in w for w in warnings), warnings


def test_invented_project_heading_dropped_but_bullets_kept():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [],
            "projects": [{
                "projectName": "Lakehouse Modernization Initiative",  # invented — not in text
                "clientName": None,
                "projectResponsibilities": ["Mentored four junior engineers on data modeling best practices"],
            }],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    job = merged["work_experience"][0]
    assert job["projects"] == []
    assert job["responsibilities"] == ["Mentored four junior engineers on data modeling best practices"]
    assert any("invented project heading" in w.lower() for w in warnings), warnings


def test_duplicate_bullets_deduped():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme",
            "responsibilities": ["Did the thing well", "Did the  thing   well", "Did another thing"],
        }],
    }
    merged, _ = ground_check(merged, RESUME_TEXT)
    assert len(merged["work_experience"][0]["responsibilities"]) == 2


def test_fabricated_metric_bullets_dropped():
    merged = {
        "personal_information": {},
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [
                # Real, grounded bullets — must be KEPT (even with a real metric).
                "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
                "Reduced nightly batch processing time by 40 percent across all regions",
                "Mentored four junior engineers on data modeling best practices",
                # AI-fabricated padding — ungrounded + impact/metric → must be DROPPED.
                "Improved release predictability by 40%",
                "Increased sprint velocity by 20%",
                "Delivered measurable cost optimization through license renegotiation and capacity right-sizing",
            ],
            "achievements": ["Accelerated time-to-market for 3 major platform launches"],
        }],
    }
    merged, warnings = ground_check(merged, RESUME_TEXT)
    resp = merged["work_experience"][0]["responsibilities"]
    assert resp == [
        "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
        "Reduced nightly batch processing time by 40 percent across all regions",
        "Mentored four junior engineers on data modeling best practices",
    ], resp
    assert merged["work_experience"][0]["achievements"] == [], merged["work_experience"][0]["achievements"]
    assert any("fabricated" in w.lower() for w in warnings), warnings


def test_coverage_detects_missed_lines():
    full = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "job_title": "Senior Data Engineer",
            "responsibilities": [
                "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow",
                "Reduced nightly batch processing time by 40 percent across all regions",
                "Mentored four junior engineers on data modeling best practices",
            ],
        }],
        "education": [{"institution_name": "State University", "degree": "Bachelor of Science in Computer Science"}],
        "personal_information": {"full_name": "John Smith", "email": ["john.smith@example.com"],
                                 "phone": ["(555) 123-4567"], "linkedin_url": "linkedin.com/in/john-smith",
                                 "address": None},
    }
    pct_full, missed_full = coverage_report(full, RESUME_TEXT)
    assert pct_full >= 80.0, (pct_full, missed_full)

    # Now delete a bullet — coverage must drop and the line must be reported.
    partial = {**full, "work_experience": [{
        **full["work_experience"][0],
        "responsibilities": full["work_experience"][0]["responsibilities"][:1],
    }]}
    pct_partial, missed = coverage_report(partial, RESUME_TEXT)
    assert pct_partial < pct_full
    assert any("Mentored four junior engineers" in m for m in missed), missed


def test_merge_recovered_is_additive_and_deduped():
    merged = {
        "work_experience": [{"company_name": "Acme Corporation", "responsibilities": ["Existing bullet one"]}],
        "education": [{"institution_name": "State University"}],
        "certifications": [],
        "skills": {"all_skills_raw": ["Python"], "other_skills": []},
    }
    recovered = {
        "work_bullets": [{
            "company_name": "Acme",
            "bullets": ["Existing bullet one", "Newly recovered bullet two"],
        }],
        "education": [
            {"institution_name": "State University"},                # dupe — skipped
            {"institution_name": "Community College", "degree": "AA"},
        ],
        "certifications": [{"name": "AWS Certified Solutions Architect"}],
        "skills": ["Python", "Terraform"],
        "professional_summary": "A summary.",
    }
    added = merge_recovered(merged, recovered)
    assert added["work_bullets"] == 1
    assert merged["work_experience"][0]["responsibilities"] == ["Existing bullet one", "Newly recovered bullet two"]
    assert added["education"] == 1 and len(merged["education"]) == 2
    assert added["certifications"] == 1
    assert added["skills"] == 1 and "Terraform" in merged["skills"]["other_skills"]
    assert merged["professional_summary"] == "A summary."

    # An existing summary is never OVERWRITTEN — the recovery pass returns the
    # missed fragment, not the whole section.
    merge_recovered(merged, {"professional_summary": "A summary."})
    assert merged["professional_summary"] == "A summary."


# ── Sections the resume does not have ───────────────────────────────────────

NO_SECTIONS_RESUME = """
Jane Doe
jane.doe@example.com

SUMMARY
Recent graduate seeking an entry-level data role.

TECHNICAL SKILLS
Python, SQL, Pandas, scikit-learn
Built a sentiment classifier for a course project

EDUCATION
Bachelor of Science in Computer Science, State University, 2025
"""


def test_certifications_dropped_when_the_resume_has_no_such_section():
    merged = {"certifications": [{"name": "Built a sentiment classifier for a course project"}]}
    merged, warnings = ground_check(merged, NO_SECTIONS_RESUME)
    assert merged["certifications"] == []
    assert any("no certifications section" in w for w in warnings), warnings


def test_work_experience_dropped_when_the_resume_has_no_experience_section():
    # The PAS / fresher case: skills and coursework promoted into a job.
    merged = {"work_experience": [{
        "company_name": "Python, SQL, Pandas",
        "responsibilities": ["Built a sentiment classifier for a course project"],
    }]}
    merged, warnings = ground_check(merged, NO_SECTIONS_RESUME)
    assert merged["work_experience"] == []
    assert any("no experience section" in w for w in warnings), warnings


def test_real_sections_are_left_alone():
    resume = """
WORK EXPERIENCE
Acme Corporation                     Jan 2020 - Present
Senior Engineer
• Built the ingestion service

CERTIFICATIONS
AWS Certified Solutions Architect
"""
    merged = {
        "work_experience": [{"company_name": "Acme Corporation",
                             "responsibilities": ["Built the ingestion service"]}],
        "certifications": [{"name": "AWS Certified Solutions Architect"}],
    }
    merged, _ = ground_check(merged, resume)
    assert len(merged["work_experience"]) == 1
    assert len(merged["certifications"]) == 1


def test_experience_heading_variants_are_all_recognised():
    # Failing to recognise one of these deletes the whole work history, so the
    # match is deliberately loose.
    for heading in [
        "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EXPERIENCE",
        "EXPERIENCE SUMMARY", "RELEVANT WORK EXPERIENCE", "IT EXPERIENCE",
        "EMPLOYMENT HISTORY", "Work History", "CAREER HISTORY",
        "ORGANISATIONAL SCAN", "Professional Background",
    ]:
        assert has_section(f"NAME\n\n{heading}\nAcme Corp", _EXPERIENCE_HEADING), heading
    assert not has_section("NAME\n\nTECHNICAL SKILLS\nPython", _EXPERIENCE_HEADING)


def test_certification_heading_variants_are_all_recognised():
    for heading in [
        "CERTIFICATIONS", "Certificates", "LICENSES", "Licences",
        "CERTIFICATIONS & LICENSES", "Professional Certifications",
        "TRAINING AND CERTIFICATIONS", "Credentials",
    ]:
        assert has_section(f"NAME\n\n{heading}\nAWS", _CERTIFICATION_HEADING), heading
    assert not has_section("NAME\n\nTECHNICAL SKILLS\nPython", _CERTIFICATION_HEADING)


def test_dropped_sections_are_excluded_from_coverage():
    resume = """
WORK EXPERIENCE
Acme Corporation
• Built the ingestion service for the payments platform

AWARDS AND RECOGNITION
• Employee of the Year for outstanding contribution to the platform team
• President's Club award for exceeding every quarterly target

EDUCATION
Bachelor of Science in Computer Science, State University
"""
    extracted = {"work_experience": [{
        "company_name": "Acme Corporation",
        "responsibilities": ["Built the ingestion service for the payments platform"],
    }], "education": [{"institution_name": "State University",
                       "degree": "Bachelor of Science in Computer Science"}]}

    # Against the raw text the awards lines read as gaps, and would be fed to
    # the recovery pass — which is how removed content comes back.
    _, missed_raw = coverage_report(extracted, resume)
    assert any("Employee of the Year" in m for m in missed_raw)

    # Against the audited text they do not exist at all.
    audited = strip_dropped_sections(resume)
    assert "Employee of the Year" not in audited
    assert "President's Club" not in audited
    assert "Built the ingestion service" in audited
    assert "State University" in audited

    pct, missed = coverage_report(extracted, audited)
    assert missed == [], missed
    assert pct == 100.0


# ── Split-bullet repair ─────────────────────────────────────────────────────

SPLIT_RESUME = """
WORK EXPERIENCE
Acme Corporation
• Developed an OCR pipeline for scanned claim forms
  Achieved 80% accuracy across the validation set
• Mentored two junior engineers
"""


def test_metric_split_into_its_own_bullet_is_rejoined():
    blocks = source_bullet_blocks(SPLIT_RESUME)
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "responsibilities": [
            "Developed an OCR pipeline for scanned claim forms",
            "Achieved 80% accuracy across the validation set",   # same source bullet
            "Mentored two junior engineers",
        ],
    }]}
    merged, warnings = ground_check(merged, SPLIT_RESUME)
    resp = merged["work_experience"][0]["responsibilities"]

    assert len(resp) == 2, resp
    assert resp[0] == blocks[0]
    assert "Achieved 80% accuracy across the validation set" in resp[0]
    assert resp[1] == "Mentored two junior engineers"
    assert any("Rejoined" in w for w in warnings), warnings


CLIENT_SPLIT_RESUME = """
WORK EXPERIENCE
Acme Consulting
Client: Globex Insurance
• Directed the claims migration
  reducing rating-related defect escapes by an estimated 25%
• Led the QA team through the release
"""


def test_metric_split_inside_a_project_block_is_rejoined():
    # The same "trailing detail becomes its own bullet" failure, but inside a
    # projects[].projectResponsibilities[] block instead of the flat list —
    # merge_split_bullets has to cover both, since consulting resumes put most
    # of their bullets under a named client/project rather than at the job level.
    blocks = source_bullet_blocks(CLIENT_SPLIT_RESUME)
    merged = {"work_experience": [{
        "company_name": "Acme Consulting",
        "responsibilities": [],
        "projects": [{
            "clientName": "Globex Insurance",
            "projectResponsibilities": [
                "Directed the claims migration",
                "reducing rating-related defect escapes by an estimated 25%",
                "Led the QA team through the release",
            ],
        }],
    }]}
    merged, warnings = ground_check(merged, CLIENT_SPLIT_RESUME)
    resp = merged["work_experience"][0]["projects"][0]["projectResponsibilities"]

    assert len(resp) == 2, resp
    assert resp[0] == blocks[0]
    assert "reducing rating-related defect escapes by an estimated 25%" in resp[0]
    assert resp[1] == "Led the QA team through the release"
    assert any("Rejoined" in w for w in warnings), warnings


# A long consulting resume stops using glyphs partway through: the last bullet
# of one job is followed by dozens of unbulleted lines belonging to the jobs
# after it. Folding those into that bullet produced a block containing another
# job's header, and merge_split_bullets then swapped real responsibilities for
# it — losing them and importing the wrong job's text in one move.
GLYPHLESS_TAIL_RESUME = """
WORK EXPERIENCE
Employer - IBM Global Services - Jun'03 - Jun'07
Client: Merck, NJ, USA
Responsibilities:
• Requirement gathering and GAP analysis
• Vendor Master maintenance
Employer - IBM Global Services - Jun'03 - Jun'07
Jan'04- Mar'05
Client: Ericsson, Stockholm, Sweden
Sr. Consultant - Production Planning
Responsibilities:
Business Process analysis
Configuration
"""


def test_a_bullet_does_not_swallow_the_next_job():
    blocks = source_bullet_blocks(GLYPHLESS_TAIL_RESUME)
    assert blocks == [
        "Requirement gathering and GAP analysis",
        "Vendor Master maintenance",
    ], blocks
    assert not any("Ericsson" in b for b in blocks)


def test_responsibilities_are_not_replaced_by_a_runaway_block():
    merged = {"work_experience": [{
        "company_name": "IBM Global Services",
        "responsibilities": [
            "Requirement gathering and GAP analysis",
            "Vendor Master maintenance",
        ],
    }]}
    merged, _ = ground_check(merged, GLYPHLESS_TAIL_RESUME)
    assert merged["work_experience"][0]["responsibilities"] == [
        "Requirement gathering and GAP analysis",
        "Vendor Master maintenance",
    ]


def test_a_rejoin_far_longer_than_its_parts_is_refused():
    # Even if two responsibilities both trace into one block, replacing them
    # with a block many times their length is not a rejoin — it is a swap.
    parts = ["Configuration and release", "Releasing the changes"]
    long_block = "Configuration and release " + ("padding text here " * 30) + "Releasing the changes"
    out, merged = merge_split_bullets(parts, [long_block], [_squash(long_block)])
    assert out == parts
    assert merged == 0


def test_separate_bullets_are_not_fused():
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "responsibilities": [
            "Developed an OCR pipeline for scanned claim forms",
            "Mentored two junior engineers",
        ],
    }]}
    merged, _ = ground_check(merged, SPLIT_RESUME)
    assert len(merged["work_experience"][0]["responsibilities"]) == 2


def test_recovered_bullet_matches_a_differently_spelled_company():
    # "Acme Corp" vs "Acme Corporation" failed the old strict-subset test and
    # the bullet was dropped on the floor.
    merged = {"work_experience": [
        {"company_name": "Acme Corporation", "responsibilities": []},
        {"company_name": "Globex International", "responsibilities": []},
    ]}
    added = merge_recovered(merged, {"work_bullets": [
        {"company_name": "Acme Corp", "bullets": ["Built the ingestion service"]},
    ]})
    assert added["work_bullets"] == 1
    assert added["unplaced_bullets"] == 0
    assert merged["work_experience"][0]["responsibilities"] == ["Built the ingestion service"]
    assert merged["work_experience"][1]["responsibilities"] == []


def test_recovered_bullet_falls_back_to_the_only_job():
    # One job on the resume — a naming mismatch cannot mean it belongs elsewhere.
    merged = {"work_experience": [{"company_name": "Acme Corporation", "responsibilities": []}]}
    added = merge_recovered(merged, {"work_bullets": [
        {"company_name": "", "bullets": ["Built the ingestion service"]},
    ]})
    assert added["work_bullets"] == 1
    assert merged["work_experience"][0]["responsibilities"] == ["Built the ingestion service"]


def test_unmatched_bullet_is_counted_rather_than_silently_dropped():
    merged = {"work_experience": [
        {"company_name": "Acme Corporation", "responsibilities": []},
        {"company_name": "Globex International", "responsibilities": []},
    ]}
    added = merge_recovered(merged, {"work_bullets": [
        {"company_name": "Initech", "bullets": ["Did a thing", "Did another thing"]},
    ]})
    assert added["work_bullets"] == 0
    assert added["unplaced_bullets"] == 2
    # Never guessed onto an unrelated employer.
    assert merged["work_experience"][0]["responsibilities"] == []
    assert merged["work_experience"][1]["responsibilities"] == []


# ── Figures borrowed from elsewhere on the resume ───────────────────────────

# This resume genuinely says 40% and 25% — in bullets of their own. A figure
# stapled onto a DIFFERENT bullet is still invented, and a check against the
# document as a whole could not see that.
BORROWED_FIGURE_RESUME = """
WORK EXPERIENCE
Acme Corporation                                Jan 2020 - Present
Delivery Manager
• Standardized the onboarding checklist across four delivery pods
• Reduced nightly batch processing time by 40% across all regions
• Ran the quarterly release calendar with the platform team
• Cut licence spend by 25% at renewal
"""


def test_impact_clause_borrowing_a_figure_from_elsewhere_is_removed():
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "responsibilities": [
            "Standardized the onboarding checklist across four delivery pods, "
            "reducing onboarding time 40%",
            "Ran the quarterly release calendar with the platform team, "
            "increasing delivery velocity 25% over three quarters",
        ],
    }]}
    merged, warnings = ground_check(merged, BORROWED_FIGURE_RESUME)
    resp = merged["work_experience"][0]["responsibilities"]

    assert resp == [
        "Standardized the onboarding checklist across four delivery pods",
        "Ran the quarterly release calendar with the platform team",
    ], resp
    assert any("invented figures" in w for w in warnings), warnings


def test_a_bullets_own_figures_are_kept():
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "responsibilities": [
            "Reduced nightly batch processing time by 40% across all regions",
            "Cut licence spend by 25% at renewal",
        ],
    }]}
    merged, warnings = ground_check(merged, BORROWED_FIGURE_RESUME)
    assert merged["work_experience"][0]["responsibilities"] == [
        "Reduced nightly batch processing time by 40% across all regions",
        "Cut licence spend by 25% at renewal",
    ]
    assert not any("invented figures" in w for w in warnings), warnings


# ── Department ──────────────────────────────────────────────────────────────

def test_department_is_dropped_when_the_resume_never_labels_one():
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "department": "Department of Workforce Development",
        "responsibilities": [],
    }]}
    merged, warnings = ground_check(merged, RESUME_TEXT)
    assert merged["work_experience"][0]["department"] is None
    assert any("Removed department" in w for w in warnings), warnings


def test_a_labelled_department_is_kept():
    resume = RESUME_TEXT.replace(
        "Client: Diligent Insurance",
        "Client: Diligent Insurance\nDepartment: Claims Technology",
    )
    merged = {"work_experience": [{
        "company_name": "Acme Corporation",
        "department": "Claims Technology",
        "responsibilities": [],
    }]}
    merged, _ = ground_check(merged, resume)
    assert merged["work_experience"][0]["department"] == "Claims Technology"


# ── Skills ──────────────────────────────────────────────────────────────────

def test_packed_skill_lines_are_taken_apart_and_prose_dropped():
    resume = RESUME_TEXT + """
TECHNICAL SKILLS
App Servers: WebSphere, WebLogic, Tomcat, JBoss
Work Authorization - US Permanent Resident (Green Card). No sponsorship required.
"""
    skills = {"other_skills": [
        "App Servers: WebSphere, WebLogic, Tomcat, JBoss",
        "Work Authorization - US Permanent Resident (Green Card). No sponsorship required.",
    ]}
    scrub_skills(skills, source_terms(resume))
    assert skills["other_skills"] == ["WebSphere", "WebLogic", "Tomcat", "JBoss"]
    # The unions are re-derived so they never disagree with the buckets.
    assert skills["all_skills_raw"] == ["WebSphere", "WebLogic", "Tomcat", "JBoss"]


def test_recovered_skill_line_is_split_rather_than_appended_whole():
    merged = {"skills": {"other_skills": [], "all_skills_raw": []}}
    added = merge_recovered(merged, {"skills": [
        "Containerization: Docker, Kubernetes, Helm · IaC: Terraform, CloudFormation",
    ]})
    assert added["skills"] == 5
    assert merged["skills"]["other_skills"] == [
        "Docker", "Kubernetes", "Helm", "Terraform", "CloudFormation",
    ]


def test_recovered_summary_line_is_appended_not_dropped():
    # The 99.3% case: the resume has a summary, one of its bullets went missing,
    # and recovery returns just that bullet. It used to be thrown away because
    # professional_summary was already non-empty.
    missing = "Technically adept software programmer with exceptional coding and documentation skills"
    merged = {"professional_summary": "• Ten years building data platforms\n• Led teams of up to 12"}

    added = merge_recovered(merged, {"professional_summary": missing})

    assert added["summary_lines"] == 1
    assert missing in merged["professional_summary"]
    # Appended in the bullet style the existing summary already uses, and the
    # lines that were already there are untouched.
    assert merged["professional_summary"].endswith(f"• {missing}")
    assert "• Ten years building data platforms" in merged["professional_summary"]
    assert len(merged["professional_summary"].split("\n")) == 3


def test_recovered_summary_already_present_is_not_duplicated():
    line = "Technically adept software programmer"
    merged = {"professional_summary": f"• Ten years of experience\n• {line} with strong documentation"}

    added = merge_recovered(merged, {"professional_summary": line})

    assert added["summary_lines"] == 0
    assert merged["professional_summary"].count(line) == 1


def test_recovered_summary_fills_an_empty_one():
    merged = {}
    added = merge_recovered(merged, {"professional_summary": "A summary."})
    assert added["summary_lines"] == 1
    assert merged["professional_summary"] == "A summary."


def test_is_grounded_short_names_strict():
    toks = set(_tokens(RESUME_TEXT))
    assert is_grounded("Diligent Insurance", toks)
    assert not is_grounded("Deloitte", toks)
    assert is_grounded("", toks)  # nothing checkable


# ── Invented-figure guard ───────────────────────────────────────────────────

FIGURE_RESUME = """
WORK EXPERIENCE
Acme Corporation                                Jan 2020 - Present
Senior Data Engineer
• Designed and implemented scalable ETL pipelines using Apache Spark and Airflow
• Cut infrastructure spend by $2 million during the platform consolidation
• Improved dashboard load times by 15% for the analytics team
"""

FIG_SOURCE = _figures(FIGURE_RESUME)
FIG_INDEX = source_line_index(FIGURE_RESUME)


def _repair(bullet):
    return repair_figures(bullet, FIG_SOURCE, FIG_INDEX)


def test_figure_key_normalises_magnitude_and_spacing():
    # "$2 million" and "$2M" are the same claim written two ways.
    assert _figure_key("$2 million") == _figure_key("$2M")
    assert _figure_key("40 %") == _figure_key("40%")
    assert _figure_key("2.50%") == _figure_key("2.5%")
    assert _figure_key("15%") != _figure_key("50%")


def test_bullet_with_source_figures_is_untouched():
    # Every figure is written in the resume — nothing to repair.
    bullet = "Improved dashboard load times by 15% for the analytics team"
    assert _repair(bullet) == bullet
    assert _repair("Cut infrastructure spend by $2 million during the platform consolidation") is not None


def test_bullet_without_figures_is_untouched():
    bullet = "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow"
    assert _repair(bullet) == bullet


def test_invented_percentage_restores_the_resume_line():
    # The model kept the real bullet and stapled a number the resume never gave.
    padded = (
        "Designed and implemented scalable ETL pipelines using Apache Spark "
        "and Airflow, improving throughput by 40%"
    )
    assert _repair(padded) == (
        "Designed and implemented scalable ETL pipelines using Apache Spark and Airflow"
    )


def test_invented_figure_is_excised_when_no_source_line_matches():
    # No resume line to fall back on, so the invented clause is cut and the
    # real head survives verbatim — no rewording, no added punctuation.
    padded = "Owned the quarterly capacity planning cycle, reducing cloud cost by 62%"
    assert _repair(padded) == "Owned the quarterly capacity planning cycle"


def test_wholly_invented_metric_bullet_is_dropped():
    # Nothing to anchor to and nothing worth keeping before the figure.
    assert _repair("Boosted revenue 300%") is None


def test_year_and_version_numbers_are_not_treated_as_figures():
    # Bare integers must not trip the guard: years, versions, team counts.
    assert _figures("Migrated to Python 3.9 in 2021 with a team of 12") == set()


# ── Technology guard ────────────────────────────────────────────────────────

TECH_RESUME = """
William Reed
WORK EXPERIENCE
Acme Corporation
Security Engineer
• Led security and access management across the payments platform
• Built reporting dashboards and automated the release pipeline
• Environment: Python, Node.js, CI/CD, C#, Go
"""

TECH_SRC = source_terms(TECH_RESUME)


def test_technology_named_in_resume_is_kept():
    for named in ("Python", "C#", "Go"):
        assert tech_is_grounded(named, TECH_SRC), named


def test_technology_inferred_from_a_duty_is_rejected():
    # The resume says "security and access management" and "reporting"; it never
    # names the products those duties imply.
    for inferred in ("IAM", "Kubernetes", "Power BI", "AWS", "Docker"):
        assert not tech_is_grounded(inferred, TECH_SRC), inferred


def test_short_acronym_is_not_matched_inside_a_word():
    # "William" contains "iam" — a substring check would wrongly keep IAM.
    assert not tech_is_grounded("IAM", TECH_SRC)


def test_spelling_variant_of_a_named_technology_survives():
    # Written "Node.js" / "CI/CD"; returned without the punctuation.
    assert tech_is_grounded("NodeJS", TECH_SRC)
    assert tech_is_grounded("CICD", TECH_SRC)


def test_ground_check_drops_inferred_job_technologies():
    merged = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "technologies_used": ["Python", "IAM", "Kubernetes", "Node.js"],
        }],
    }
    merged, warnings = ground_check(merged, TECH_RESUME)
    kept = merged["work_experience"][0]["technologies_used"]
    assert kept == ["Python", "Node.js"]
    assert any("not named in the resume" in w for w in warnings)


def test_scrub_skills_drops_inferred_skills_across_buckets():
    skills = {
        "programming_languages": ["Python", "Ruby"],
        "cloud_platforms": ["AWS"],
        "categories": [{"name": "Security", "skills": ["IAM", "C#"]}],
    }
    dropped = scrub_skills(skills, TECH_SRC)
    assert skills["programming_languages"] == ["Python"]
    assert skills["cloud_platforms"] == []
    assert skills["categories"][0]["skills"] == ["C#"]
    assert dropped == 3


def test_ground_check_strips_invented_percentage_from_a_job():
    merged = {
        "work_experience": [{
            "company_name": "Acme Corporation",
            "responsibilities": [
                "Designed and implemented scalable ETL pipelines using Apache "
                "Spark and Airflow, improving throughput by 40%",
                "Improved dashboard load times by 15% for the analytics team",
            ],
        }],
    }
    merged, warnings = ground_check(merged, FIGURE_RESUME)
    resp = merged["work_experience"][0]["responsibilities"]

    # The invented 40% is gone; the real 15% the resume states is preserved.
    assert not any("40%" in r for r in resp)
    assert any("15%" in r for r in resp)
    assert len(resp) == 2  # the real bullet was repaired, not dropped
    assert any("invented figures" in w for w in warnings)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok: {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL: {name}: {exc}")
    if failures:
        sys.exit(1)
    print("all auditor tests passed")
