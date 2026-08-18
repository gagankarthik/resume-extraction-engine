"""
Tests for the deterministic shape pass: degrees, names, credentials, current
roles, and the skills tidier.

No API keys or network needed. Run directly:  python tests/test_polish.py
(also compatible with pytest if installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.degrees import abbreviate, normalize_education, split_degree
from agents.polish import (
    clean_name,
    dedupe_against_certifications,
    dedupe_roles,
    polish,
    polish_certifications,
    polish_personal,
    polish_work,
    strip_label,
    tidy_date,
    titlecase_name,
)
from agents.skills import looks_like_skill, split_packed_skills, tidy_skill_list

# ── Degrees ─────────────────────────────────────────────────────────────────

def test_written_out_degrees_become_abbreviations():
    cases = {
        "Bachelor's degree in Accounting & Business Management": "BS",
        "Associate's degree in computer science & Programming": "AS",
        "Master's degree in Information Systems": "MS",
        "Bachelor of Science in Computer Science": "BS",
        "Master of Business Administration": "MBA",
        "Bachelor of Technology in Electronics": "BTech",
        "Bachelor of Engineering": "BE",
        "Master of Computer Applications": "MCA",
        "Doctor of Philosophy in Statistics": "PhD",
        "Post Graduate Diploma in Management": "PGDM",
    }
    for written, expected in cases.items():
        assert split_degree(written)[0] == expected, (written, split_degree(written))


def test_families_are_never_swapped():
    assert abbreviate("B.Tech") == "BTech"
    assert abbreviate("Bachelor of Science") == "BS"
    assert abbreviate("B.S.") == "BS"
    assert abbreviate("BSc") == "BS"
    assert abbreviate("M.S.") == "MS"
    assert abbreviate("MBA") == "MBA"
    assert abbreviate("Nothing degree-like here") is None


def test_subject_moves_into_field_of_study():
    edu = [{
        "institution_name": "State University",
        "degree": "Bachelor's degree in Accounting & Business Management",
        "degree_type": None,
        "field_of_study": None,
    }]
    assert normalize_education(edu) == 1
    assert edu[0]["degree"] == "BS"
    assert edu[0]["degree_type"] == "BS"
    assert edu[0]["field_of_study"] == "Accounting & Business Management"


def test_existing_field_of_study_is_not_overwritten():
    edu = [{
        "degree": "Bachelor of Science in CS",
        "field_of_study": "Computer Science",
    }]
    normalize_education(edu)
    assert edu[0]["degree"] == "BS"
    assert edu[0]["field_of_study"] == "Computer Science"


def test_already_abbreviated_degree_is_left_alone():
    edu = [{"degree": "BS", "degree_type": "BS", "field_of_study": "Physics"}]
    assert normalize_education(edu) == 0
    assert edu[0]["degree"] == "BS"


# ── Names ───────────────────────────────────────────────────────────────────

def test_nickname_in_brackets_is_dropped():
    assert clean_name("BalaKrishna (Krishna) Jallipalli") == "BalaKrishna Jallipalli"
    assert clean_name('Robert "Bob" Smith') == "Robert Smith"


def test_ordinary_names_survive_untouched():
    for name in ("Siobhan O'Brien", "Mary Smith-Jones", "John Smith Jr.", "JOHN SMITH"):
        assert clean_name(name) == name


def test_name_parts_are_filled_from_the_cleaned_full_name():
    pi = {"full_name": "BalaKrishna (Krishna) Jallipalli", "first_name": None, "last_name": None}
    polish_personal(pi)
    assert pi["full_name"] == "BalaKrishna Jallipalli"
    assert pi["first_name"] == "BalaKrishna"
    assert pi["last_name"] == "Jallipalli"


def test_full_name_assembled_when_only_parts_survive():
    pi = {"full_name": None, "first_name": "Ada", "last_name": "Lovelace"}
    polish_personal(pi)
    assert pi["full_name"] == "Ada Lovelace"


def test_shouted_name_is_recased_to_title_case():
    assert titlecase_name("SHASHE KIRAN GANJI") == "Shashe Kiran Ganji"
    assert titlecase_name("shashe kiran ganji") == "Shashe Kiran Ganji"


def test_mixed_case_name_is_left_alone():
    # Written this way on purpose — a blind title-case would flatten it.
    for name in ("Siobhan O'Brien", "Mary Smith-Jones", "McDonald", "DeSouza"):
        assert titlecase_name(name) == name


def test_polish_personal_recases_a_shouted_name():
    pi = {"full_name": "SHASHE KIRAN GANJI", "first_name": "SHASHE KIRAN", "last_name": "GANJI"}
    polish_personal(pi)
    assert pi["full_name"] == "Shashe Kiran Ganji"
    assert pi["first_name"] == "Shashe Kiran"
    assert pi["last_name"] == "Ganji"


# ── Labels ──────────────────────────────────────────────────────────────────

def test_inline_labels_are_stripped_from_values():
    assert strip_label("Client: Department of Workforce Development") == \
        "Department of Workforce Development"
    assert strip_label("End Client - Acme Insurance") == "Acme Insurance"
    assert strip_label("Acme Insurance") == "Acme Insurance"


# ── Current roles ───────────────────────────────────────────────────────────

def test_current_roles_all_read_till_date():
    work = [
        {"company_name": "Acme", "end_date": "Present", "is_current": False},
        {"company_name": "Beta", "end_date": "Current", "is_current": False},
        {"company_name": "Gamma", "end_date": "Dec 2019", "is_current": False},
        {"company_name": "Delta", "end_date": None, "is_current": True},
    ]
    polish_work(work)
    assert work[0]["end_date"] == "Till Date" and work[0]["is_current"] is True
    assert work[1]["end_date"] == "Till Date" and work[1]["is_current"] is True
    assert work[2]["end_date"] == "Dec 2019" and work[2]["is_current"] is False
    assert work[3]["end_date"] == "Till Date"


def test_date_scaffolding_is_taken_off():
    assert tidy_date("Since June 2025 -") == "June 2025"
    assert tidy_date("Nov - 2010") == "Nov 2010"
    assert tidy_date("From Jan 2020") == "Jan 2020"
    # Ordinary dates are left exactly as the resume wrote them.
    assert tidy_date("Feb' 09") == "Feb' 09"
    assert tidy_date("Jan 2020") == "Jan 2020"


def test_a_start_date_written_as_a_sentence_is_cleaned_and_marked_current():
    work = [{"company_name": "Acme", "start_date": "Since June 2025 -", "end_date": None,
             "is_current": True}]
    polish_work(work)
    assert work[0]["start_date"] == "June 2025"
    assert work[0]["end_date"] == "Till Date"


def test_client_bullets_are_not_repeated_in_the_flat_list():
    work = [{
        "company_name": "Client: Acme Consulting",
        "responsibilities": ["Led the claims migration", "Ran the daily standup"],
        "projects": [{
            "clientName": "Client: Diligent Insurance",
            "projectResponsibilities": ["Led the claims migration"],
        }],
    }]
    polish_work(work)
    assert work[0]["company_name"] == "Acme Consulting"
    assert work[0]["projects"][0]["clientName"] == "Diligent Insurance"
    assert work[0]["responsibilities"] == ["Ran the daily standup"]


def test_project_technologies_are_not_repeated_at_the_job_level():
    # A job with one project has technologies_used identical to that project's
    # keyTechnologies — printing both shows "Key Technologies/Skills" twice,
    # word for word, once under the job and once under the project.
    work = [{
        "company_name": "Capgemini USA Inc.",
        "technologies_used": ["TOSCA", "Guidewire PolicyCenter"],
        "projects": [{
            "clientName": "Grange Insurance",
            "keyTechnologies": "TOSCA, Guidewire PolicyCenter",
        }],
    }]
    polish_work(work)
    assert work[0]["technologies_used"] == []
    assert work[0]["projects"][0]["keyTechnologies"] == "TOSCA, Guidewire PolicyCenter"


def test_a_job_level_technology_outside_any_project_survives():
    work = [{
        "company_name": "Capgemini USA Inc.",
        "technologies_used": ["TOSCA", "Jira"],
        "projects": [{
            "clientName": "Grange Insurance",
            "keyTechnologies": "TOSCA",
        }],
    }]
    polish_work(work)
    assert work[0]["technologies_used"] == ["Jira"]


# ── One role, listed once ───────────────────────────────────────────────────

def test_the_same_engagement_listed_twice_is_kept_once():
    """A summary table and a detail section describing the same work.

    The fuller entry survives; anything only the short one knew comes with it.
    """
    work = [
        {"company_name": "Boeing", "start_date": "Dec 2023", "end_date": "May 2025",
         "location": "Seattle, WA", "responsibilities": []},
        {"company_name": "Boeing", "start_date": "Dec 2023", "end_date": "May 2025",
         "location": None, "job_title": "SAP Solution Architect",
         "responsibilities": ["Solution design for Global Services projects", "Estimation"]},
    ]
    assert dedupe_roles(work) == 1
    assert len(work) == 1
    kept = work[0]
    assert len(kept["responsibilities"]) == 2
    assert kept["job_title"] == "SAP Solution Architect"
    # The location only the summary row carried is not lost with it.
    assert kept["location"] == "Seattle, WA"


def test_separate_stints_at_one_employer_are_all_kept():
    """This candidate really did return to Cardinal Health five times."""
    work = [
        {"company_name": "Cardinal Health", "start_date": "Dec 2022", "responsibilities": ["a"]},
        {"company_name": "Cardinal Health", "start_date": "July 2019", "responsibilities": ["b"]},
        {"company_name": "Cardinal Health", "start_date": "Sep 2017", "responsibilities": ["c"]},
    ]
    assert dedupe_roles(work) == 0
    assert len(work) == 3


def test_a_role_with_no_date_is_never_merged_away():
    # Without a start date there is not enough identity to call it a duplicate.
    work = [
        {"company_name": "Acme", "start_date": None, "responsibilities": ["a"]},
        {"company_name": "Acme", "start_date": None, "responsibilities": ["b"]},
    ]
    assert dedupe_roles(work) == 0
    assert len(work) == 2


def test_document_order_is_preserved_when_a_duplicate_wins():
    work = [
        {"company_name": "Alpha", "start_date": "2020", "responsibilities": ["x"]},
        {"company_name": "Beta", "start_date": "2019", "responsibilities": []},
        {"company_name": "Gamma", "start_date": "2018", "responsibilities": ["z"]},
        {"company_name": "Beta", "start_date": "2019", "responsibilities": ["y1", "y2"]},
    ]
    dedupe_roles(work)
    assert [j["company_name"] for j in work] == ["Alpha", "Beta", "Gamma"]
    assert work[1]["responsibilities"] == ["y1", "y2"]


# ── Credentials ─────────────────────────────────────────────────────────────

def test_issuer_is_split_off_the_certification_name():
    certs = [{"name": "Certified Agile Scrum Master - Scrum Alliance, USA",
              "issuing_organization": None}]
    polish_certifications(certs)
    assert certs[0]["name"] == "Certified Agile Scrum Master"
    assert certs[0]["issuing_organization"] == "Scrum Alliance, USA"


def test_the_same_credential_twice_is_listed_once():
    certs = [
        {"name": "Certified Software Tester (CSTE) - Quality Assurance Institute (QAI), USA"},
        {"name": "Certified Software Tester (CSTE) — Quality Assurance Institute (QAI), USA"},
    ]
    assert polish_certifications(certs) == 1
    assert len(certs) == 1


def test_training_repeating_a_certification_is_dropped():
    merged = {
        "certifications": [
            {"name": "Certified Agile Scrum Master", "issuing_organization": "Scrum Alliance, USA"},
            {"name": "Azure DevOps Fundamentals", "issuing_organization": "Udemy"},
        ],
        "training": [
            {"name": "Certified Agile Scrum Master", "provider": "Scrum Alliance, USA"},
            {"name": "Internal Leadership Program", "provider": "Capgemini University"},
        ],
        "courses": [{"name": "Azure DevOps Fundamentals", "provider": "Udemy"}],
    }
    assert dedupe_against_certifications(merged) == 2
    assert [t["name"] for t in merged["training"]] == ["Internal Leadership Program"]
    assert merged["courses"] == []


# ── Skills ──────────────────────────────────────────────────────────────────

def test_packed_category_lines_become_individual_skills():
    line = "Containerization: Docker, Kubernetes, Helm, Ansible · IaC: Terraform, CloudFormation, CDK"
    assert split_packed_skills(line) == [
        "Docker", "Kubernetes", "Helm", "Ansible",
        "Terraform", "CloudFormation", "CDK",
    ]


def test_prose_is_not_a_skill():
    assert not looks_like_skill(
        "Work Authorization - US Permanent Resident (Green Card). No sponsorship required."
    )
    assert not looks_like_skill("12+ years of experience delivering enterprise programs")
    assert looks_like_skill("Kubernetes")
    assert looks_like_skill("CI/CD")
    assert looks_like_skill("FedRAMP-adjacent patterns")


def test_a_skills_bucket_is_exploded_deduped_and_cleaned():
    tidied, split, dropped = tidy_skill_list([
        "App Servers: WebSphere, WebLogic, Tomcat, JBoss",
        "Work Authorization - US Permanent Resident (Green Card). No sponsorship required.",
        "Tomcat",
        "Kubernetes",
    ])
    assert tidied == ["WebSphere", "WebLogic", "Tomcat", "JBoss", "Kubernetes"]
    assert split == 1
    assert dropped == 1


def test_single_skills_pass_through_unchanged():
    tidied, split, dropped = tidy_skill_list(["Python", "Apache Spark", "UI/UX"])
    assert tidied == ["Python", "Apache Spark", "UI/UX"]
    assert (split, dropped) == (0, 0)


def test_punctuation_inside_brackets_is_part_of_the_name():
    # "Plan to produce (PTM PP, PP-PI)" is one skill the resume named. Splitting
    # on that comma leaves the reader with "PP-PI)" as a technology.
    for skill in (
        "Plan to produce (PTM PP, PP-PI)",
        "SAP APO (DP/SNP)",
        "FI - Invoice Processing (Accounts Payable PTP and Receivable OTC)",
    ):
        assert split_packed_skills(skill) == [skill], skill


def test_bracketed_names_survive_a_packed_line():
    line = "SAP: Procure to Pay (PTP), Plan to produce (PTM PP, PP-PI), QM"
    assert split_packed_skills(line) == [
        "Procure to Pay (PTP)", "Plan to produce (PTM PP, PP-PI)", "QM",
    ]


# ── End to end ──────────────────────────────────────────────────────────────

def test_polish_reports_what_it_changed():
    merged = {
        "personal_information": {"full_name": "BalaKrishna (Krishna) Jallipalli"},
        "education": [{"degree": "Master's degree in Information Systems"}],
        "certifications": [
            {"name": "Certified Agile Scrum Master - Scrum Alliance, USA"},
            {"name": "Certified Agile Scrum Master — Scrum Alliance, USA"},
        ],
        "training": [{"name": "Certified Agile Scrum Master", "provider": "Scrum Alliance, USA"}],
        "work_experience": [{"company_name": "Acme", "end_date": "Present"}],
    }
    notes = polish(merged)
    assert merged["personal_information"]["full_name"] == "BalaKrishna Jallipalli"
    assert merged["education"][0]["degree"] == "MS"
    assert len(merged["certifications"]) == 1
    assert merged["training"] == []
    assert merged["work_experience"][0]["end_date"] == "Till Date"
    assert len(notes) == 3, notes


def test_polish_never_raises_on_junk():
    assert polish({}) == []
    assert polish({"education": None, "certifications": "nope", "work_experience": 7}) == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
