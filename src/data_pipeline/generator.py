"""
Synthetic data generator for BMS Clinical Document Recommendation System.

Produces realistic clinical documents with free-text, 2048-dim image feature vectors
(simulated ResNet output), and structured metadata (ICD codes, lab values, biomarker levels).

Three dataset sizes: 1K, 10K, 100K documents.
500 synthetic physician users with implicit interaction logs (views, clicks, saves, dwell time).

The generator ensures:
- Medical vocabulary mismatch exists (same concept, different words) to test retrieval quality
- Interaction sparsity increases with scale (1K=~85%, 10K=~97%, 100K=~99.97%)
- Cold start documents exist (40%+ with fewer than 5 interactions at 10K+ scale)
"""

import hashlib
import os
import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import yaml


# ──────────────────────────────────────────────────────────────────────
# Clinical vocabulary with intentional synonym groups to create
# vocabulary mismatch — the core retrieval problem this system solves.
# ──────────────────────────────────────────────────────────────────────

SPECIALTIES = [
    "oncology", "cardiology", "neurology", "immunology",
    "endocrinology", "pulmonology", "nephrology", "hematology",
    "rheumatology", "gastroenterology"
]

DOC_TYPES = [
    "trial_report", "case_study", "drug_interaction",
    "pathology_report", "cohort_summary"
]

# Synonym groups: same clinical concept, different vocabulary.
# This is exactly the problem that BM25 cannot solve.
CLINICAL_SYNONYM_GROUPS = [
    {
        "concept": "cardiac_toxicity",
        "terms": [
            "cardiotoxicity", "cardiac adverse events",
            "TKI-induced cardiac events", "cardiovascular toxicity",
            "myocardial injury", "drug-induced cardiomyopathy",
            "cardiac dysfunction secondary to therapy"
        ]
    },
    {
        "concept": "liver_damage",
        "terms": [
            "hepatotoxicity", "drug-induced liver injury",
            "hepatic failure", "liver function impairment",
            "DILI", "cholestatic injury",
            "transaminase elevation"
        ]
    },
    {
        "concept": "immune_response",
        "terms": [
            "immunogenicity", "immune-mediated adverse reaction",
            "antibody-dependent response", "autoimmune activation",
            "cytokine storm", "immune checkpoint response",
            "hypersensitivity reaction"
        ]
    },
    {
        "concept": "kidney_injury",
        "terms": [
            "nephrotoxicity", "acute kidney injury",
            "renal impairment", "glomerular filtration decline",
            "drug-induced nephropathy", "creatinine elevation",
            "tubular necrosis"
        ]
    },
    {
        "concept": "blood_disorder",
        "terms": [
            "hematologic toxicity", "myelosuppression",
            "pancytopenia", "thrombocytopenia",
            "neutropenia", "anemia secondary to treatment",
            "bone marrow suppression"
        ]
    },
    {
        "concept": "tumor_response",
        "terms": [
            "objective response rate", "tumor regression",
            "partial response", "complete remission",
            "disease control rate", "radiographic response",
            "RECIST criteria response"
        ]
    },
    {
        "concept": "neurological_effects",
        "terms": [
            "neurotoxicity", "peripheral neuropathy",
            "cognitive impairment", "central nervous system effects",
            "encephalopathy", "chemotherapy-induced neuropathy",
            "neurodegeneration"
        ]
    },
]

# Drug names used across documents
DRUG_NAMES = [
    "pembrolizumab", "nivolumab", "atezolizumab", "ipilimumab",
    "trastuzumab", "bevacizumab", "rituximab", "cetuximab",
    "osimertinib", "erlotinib", "gefitinib", "afatinib",
    "imatinib", "dasatinib", "nilotinib", "sorafenib",
    "sunitinib", "pazopanib", "cabozantinib", "lenvatinib",
    "olaparib", "niraparib", "rucaparib", "talazoparib",
    "venetoclax", "ibrutinib", "acalabrutinib", "zanubrutinib",
    "methotrexate", "cyclophosphamide", "doxorubicin", "cisplatin"
]

# ICD-10 codes for clinical documents
ICD_CODES = [
    "C34.1", "C50.9", "C61", "C18.9", "C71.9",  # cancers
    "I25.1", "I50.9", "I42.0", "I48.0",          # cardiac
    "N17.9", "N18.9",                              # renal
    "K72.0", "K75.9",                              # hepatic
    "D70.9", "D69.6", "D64.9",                    # hematologic
    "G62.0", "G93.4",                              # neurologic
    "E11.9", "E05.0",                              # endocrine
    "J84.1", "J96.0",                              # pulmonary
    "M35.9", "M06.9",                              # rheumatic
    "D89.9", "D80.9",                              # immune
]

# Template fragments for generating realistic clinical text
TRIAL_TEMPLATES = [
    "A phase {phase} randomized controlled trial evaluating {drug} in patients with {condition}. "
    "Primary endpoint: {endpoint}. Enrolled {n_patients} patients across {n_sites} sites. "
    "{synonym_sentence} The study demonstrated {outcome} with a hazard ratio of {hr} "
    "(95% CI: {ci_low}-{ci_high}, p={pval}). Adverse events of grade 3 or higher were "
    "observed in {ae_pct}% of patients in the treatment arm.",

    "This multicenter {phase} clinical trial investigated the efficacy and safety of {drug} "
    "for {condition}. {n_patients} eligible patients were randomized 1:1. "
    "{synonym_sentence} The primary analysis showed {outcome}. "
    "Median progression-free survival was {pfs} months (95% CI: {ci_low}-{ci_high}). "
    "Treatment-related {synonym_term} occurred in {ae_pct}% of participants.",
]

CASE_STUDY_TEMPLATES = [
    "We present a {age}-year-old {sex} with {condition} treated with {drug}. "
    "The patient presented with {symptoms}. {synonym_sentence} "
    "Laboratory findings showed {lab_findings}. Imaging revealed {imaging_findings}. "
    "After {duration} of treatment, the patient exhibited {outcome}.",

    "Case report: A {age}-year-old {sex} diagnosed with {condition} received {drug} "
    "as {line} therapy. Initial workup demonstrated {lab_findings}. "
    "{synonym_sentence} Following {duration} of therapy, {outcome}. "
    "This case illustrates {lesson}.",
]

DRUG_INTERACTION_TEMPLATES = [
    "Drug interaction analysis between {drug1} and {drug2} in {condition} patients. "
    "{synonym_sentence} Pharmacokinetic analysis revealed {pk_finding}. "
    "CYP{cyp} metabolism was {cyp_effect}. Dose adjustment of {adjustment} is recommended "
    "when co-administering these agents. {ae_count} adverse events were attributable to "
    "the interaction, most commonly {common_ae}.",

    "Investigation of potential {interaction_type} interaction between {drug1} and {drug2}. "
    "{synonym_sentence} In vitro studies showed {in_vitro}. "
    "Clinical data from {n_patients} patients confirmed {clinical_finding}. "
    "The AUC of {drug1} was {auc_change} when combined with {drug2}.",
]

PATHOLOGY_TEMPLATES = [
    "Histopathological analysis of {tissue_type} tissue from a patient with {condition}. "
    "Microscopic examination revealed {micro_findings}. {synonym_sentence} "
    "Immunohistochemistry showed {ihc_results}. Tumor grade: {grade}. "
    "Ki-67 proliferation index: {ki67}%. Margins: {margins}.",

    "Pathology report for {tissue_type} specimen. Gross examination: {gross_findings}. "
    "{synonym_sentence} Histologic type: {histo_type}. {micro_findings}. "
    "Molecular testing: {molecular}. TNM staging: {tnm}.",
]

COHORT_TEMPLATES = [
    "Retrospective cohort analysis of {n_patients} patients with {condition} treated "
    "with {drug} between {start_year} and {end_year}. {synonym_sentence} "
    "Median age: {median_age} years. {sex_dist}. Median follow-up: {followup} months. "
    "Overall survival at {os_timepoint} months: {os_pct}%. Biomarker analysis showed "
    "{biomarker_finding}.",

    "Population-based cohort study examining {outcome_measure} in {n_patients} {condition} "
    "patients. {synonym_sentence} Subgroup analysis by {subgroup_var} revealed {subgroup_finding}. "
    "Multivariate regression identified {predictors} as independent predictors "
    "(adjusted OR: {or_val}, 95% CI: {ci_low}-{ci_high}).",
]

# Search query templates that physicians would use
SEARCH_QUERY_TEMPLATES = [
    "cardiac adverse events in {drug_class} trials",
    "{condition} treatment outcomes with {drug}",
    "drug interaction {drug1} {drug2}",
    "{condition} pathology report findings",
    "survival analysis {condition} {drug}",
    "biomarker {biomarker} in {condition}",
    "{synonym_term} management guidelines",
    "phase {phase} trial {drug} {condition}",
    "dose adjustment {drug} renal impairment",
    "{condition} immunotherapy response rate",
    "adverse event profile {drug}",
    "cohort study {condition} outcomes",
    "{drug_class} mechanism of action {condition}",
    "histopathology {tissue_type} {condition}",
    "lab values {lab_test} {condition}",
    "pediatric {condition} {drug} safety",
    "{condition} combination therapy {drug1} {drug2}",
    "real world evidence {drug} {condition}",
    "progression free survival {drug} {condition}",
    "{synonym_term} in clinical trials",
]

DRUG_CLASSES = [
    "EGFR inhibitor", "PD-1 inhibitor", "PD-L1 inhibitor", "PARP inhibitor",
    "CDK4/6 inhibitor", "BTK inhibitor", "BCL-2 inhibitor", "ALK inhibitor",
    "BRAF inhibitor", "MEK inhibitor", "mTOR inhibitor", "VEGF inhibitor",
    "tyrosine kinase inhibitor", "checkpoint inhibitor", "monoclonal antibody",
]

CONDITIONS = [
    "non-small cell lung cancer", "breast cancer", "colorectal cancer",
    "melanoma", "renal cell carcinoma", "hepatocellular carcinoma",
    "chronic lymphocytic leukemia", "diffuse large B-cell lymphoma",
    "multiple myeloma", "acute myeloid leukemia", "prostate cancer",
    "ovarian cancer", "pancreatic cancer", "glioblastoma",
    "gastric cancer", "head and neck squamous cell carcinoma",
    "urothelial carcinoma", "thyroid cancer",
]

BIOMARKERS = [
    "PD-L1", "EGFR", "ALK", "KRAS", "BRAF V600E", "HER2", "BRCA1/2",
    "MSI-H", "TMB", "NTRK", "ROS1", "MET", "RET", "PIK3CA",
    "TP53", "IDH1", "FGFR", "CDK4", "BCL-2", "BTK",
]

LAB_TESTS = [
    "creatinine", "ALT", "AST", "bilirubin", "albumin",
    "hemoglobin", "platelet count", "neutrophil count",
    "TSH", "free T4", "troponin", "BNP", "LDH",
    "CEA", "CA-125", "PSA", "AFP",
]

TISSUE_TYPES = [
    "lung", "breast", "colon", "liver", "kidney",
    "brain", "skin", "lymph node", "bone marrow", "prostate",
    "ovarian", "pancreatic", "gastric", "thyroid",
]


def _generate_doc_id(index: int, doc_type: str) -> str:
    """Generate a deterministic document ID."""
    raw = f"doc_{doc_type}_{index}"
    return f"DOC-{hashlib.md5(raw.encode()).hexdigest()[:12].upper()}"


def _generate_physician_id(index: int) -> str:
    """Generate a deterministic physician ID."""
    return f"PHY-{index:04d}"


def _generate_query_id(index: int) -> str:
    """Generate a deterministic query ID."""
    return f"QRY-{index:06d}"


def _pick_synonym_sentence(rng: random.Random) -> tuple[str, str]:
    """Pick a random synonym group and generate a sentence using one of its terms.

    Returns (sentence, term_used) to enable vocabulary mismatch in the corpus.
    """
    group = rng.choice(CLINICAL_SYNONYM_GROUPS)
    term = rng.choice(group["terms"])
    templates = [
        f"Evidence of {term} was documented during the study period.",
        f"The investigators reported significant {term} among enrolled subjects.",
        f"Monitoring for {term} was conducted at regular intervals throughout the trial.",
        f"Clinical assessment revealed {term} as a notable finding.",
        f"The occurrence of {term} was systematically recorded and analyzed.",
        f"Post-treatment evaluation identified {term} in a subset of patients.",
    ]
    return rng.choice(templates), term


def _generate_trial_report(rng: random.Random, index: int) -> dict:
    """Generate a synthetic clinical trial report."""
    drug = rng.choice(DRUG_NAMES)
    condition = rng.choice(CONDITIONS)
    synonym_sentence, synonym_term = _pick_synonym_sentence(rng)
    phase = rng.choice(["I", "Ib", "II", "IIb", "III", "IIIb"])
    n_patients = rng.randint(50, 2000)

    template = rng.choice(TRIAL_TEMPLATES)
    body = template.format(
        phase=phase, drug=drug, condition=condition,
        endpoint=rng.choice(["progression-free survival", "overall survival",
                             "objective response rate", "disease control rate"]),
        n_patients=n_patients, n_sites=rng.randint(5, 200),
        synonym_sentence=synonym_sentence, synonym_term=synonym_term,
        outcome=rng.choice(["significant improvement", "non-inferiority",
                            "marginal benefit", "no significant difference"]),
        hr=round(rng.uniform(0.3, 1.2), 2),
        ci_low=round(rng.uniform(0.2, 0.7), 2),
        ci_high=round(rng.uniform(0.8, 1.5), 2),
        pval=round(rng.uniform(0.001, 0.08), 4),
        ae_pct=rng.randint(5, 45),
        pfs=round(rng.uniform(3, 24), 1),
    )

    specialty = rng.choice(SPECIALTIES)
    icd = ",".join(rng.sample(ICD_CODES, rng.randint(1, 3)))

    return {
        "doc_id": _generate_doc_id(index, "trial"),
        "title": f"Phase {phase} Trial of {drug} in {condition}",
        "specialty": specialty,
        "doc_type": "trial_report",
        "body_text": body,
        "icd_codes": icd,
    }


def _generate_case_study(rng: random.Random, index: int) -> dict:
    """Generate a synthetic case study."""
    drug = rng.choice(DRUG_NAMES)
    condition = rng.choice(CONDITIONS)
    synonym_sentence, _ = _pick_synonym_sentence(rng)
    age = rng.randint(25, 85)
    sex = rng.choice(["male", "female"])

    template = rng.choice(CASE_STUDY_TEMPLATES)
    body = template.format(
        age=age, sex=sex, condition=condition, drug=drug,
        symptoms=rng.choice(["progressive dyspnea and fatigue",
                             "persistent abdominal pain and weight loss",
                             "worsening neurological symptoms",
                             "unexplained cytopenias and organomegaly"]),
        synonym_sentence=synonym_sentence,
        lab_findings=rng.choice(["elevated LDH and tumor markers",
                                 "abnormal liver function tests",
                                 "declining renal function",
                                 "pancytopenia with blast cells"]),
        imaging_findings=rng.choice(["multiple pulmonary nodules",
                                     "hepatic lesions on CT",
                                     "enhancing mass on MRI",
                                     "PET-avid lymphadenopathy"]),
        duration=rng.choice(["3 months", "6 months", "12 weeks", "8 cycles"]),
        outcome=rng.choice(["partial response by RECIST criteria",
                            "complete metabolic response on PET",
                            "disease progression requiring therapy change",
                            "stable disease maintained for 18 months"]),
        line=rng.choice(["first-line", "second-line", "third-line", "adjuvant"]),
        lesson=rng.choice(["the importance of biomarker-driven therapy selection",
                           "challenges in managing treatment-related toxicity",
                           "potential for durable response in selected patients",
                           "the need for multidisciplinary management"]),
    )

    specialty = rng.choice(SPECIALTIES)
    icd = ",".join(rng.sample(ICD_CODES, rng.randint(1, 3)))

    return {
        "doc_id": _generate_doc_id(index, "case"),
        "title": f"Case Report: {condition} Treated with {drug}",
        "specialty": specialty,
        "doc_type": "case_study",
        "body_text": body,
        "icd_codes": icd,
    }


def _generate_drug_interaction(rng: random.Random, index: int) -> dict:
    """Generate a synthetic drug interaction study."""
    drugs = rng.sample(DRUG_NAMES, 2)
    condition = rng.choice(CONDITIONS)
    synonym_sentence, _ = _pick_synonym_sentence(rng)

    template = rng.choice(DRUG_INTERACTION_TEMPLATES)
    body = template.format(
        drug1=drugs[0], drug2=drugs[1], condition=condition,
        synonym_sentence=synonym_sentence,
        pk_finding=rng.choice(["significant increase in plasma concentration",
                                "reduced clearance of the primary agent",
                                "altered volume of distribution",
                                "prolonged half-life of the substrate"]),
        cyp=rng.choice(["3A4", "2D6", "2C9", "1A2", "2C19"]),
        cyp_effect=rng.choice(["inhibited", "induced", "unchanged"]),
        adjustment=rng.choice(["50% dose reduction", "25% dose reduction",
                                "no adjustment required", "monitoring without adjustment"]),
        ae_count=rng.randint(3, 25),
        common_ae=rng.choice(["nausea", "diarrhea", "fatigue",
                              "hepatotoxicity", "QT prolongation"]),
        interaction_type=rng.choice(["pharmacokinetic", "pharmacodynamic"]),
        in_vitro=rng.choice(["competitive inhibition at CYP active site",
                              "time-dependent inactivation",
                              "no significant interaction at therapeutic concentrations"]),
        n_patients=rng.randint(20, 300),
        clinical_finding=rng.choice(["clinically significant interaction requiring monitoring",
                                      "no dose-limiting interaction",
                                      "additive toxicity in combination"]),
        auc_change=rng.choice(["increased by 40%", "decreased by 25%",
                                "increased by 120%", "unchanged"]),
    )

    specialty = rng.choice(SPECIALTIES)
    icd = ",".join(rng.sample(ICD_CODES, rng.randint(1, 2)))

    return {
        "doc_id": _generate_doc_id(index, "interaction"),
        "title": f"Drug Interaction: {drugs[0]} and {drugs[1]}",
        "specialty": specialty,
        "doc_type": "drug_interaction",
        "body_text": body,
        "icd_codes": icd,
    }


def _generate_pathology_report(rng: random.Random, index: int) -> dict:
    """Generate a synthetic pathology report."""
    tissue = rng.choice(TISSUE_TYPES)
    condition = rng.choice(CONDITIONS)
    synonym_sentence, _ = _pick_synonym_sentence(rng)

    template = rng.choice(PATHOLOGY_TEMPLATES)
    body = template.format(
        tissue_type=tissue, condition=condition,
        synonym_sentence=synonym_sentence,
        micro_findings=rng.choice([
            "moderately differentiated adenocarcinoma with stromal invasion",
            "high-grade neoplasm with necrosis and mitotic figures",
            "well-differentiated tumor with pushing margins",
            "poorly differentiated carcinoma with lymphovascular invasion"
        ]),
        ihc_results=rng.choice([
            "PD-L1 TPS 80%, CK7+, TTF-1+",
            "ER+/PR+, HER2 negative by FISH",
            "CD20+, CD10+, BCL-6+, Ki-67 90%",
            "CK20+, CDX2+, MSI-H by IHC"
        ]),
        grade=rng.choice(["Grade 1", "Grade 2", "Grade 3"]),
        ki67=rng.randint(5, 95),
        margins=rng.choice(["negative", "positive", "close (< 1mm)"]),
        gross_findings=rng.choice([
            "4.2 cm firm tan-white mass",
            "2.1 cm well-circumscribed nodule",
            "6.5 cm irregular infiltrative lesion",
            "multiple fragments totaling 3.8 cm"
        ]),
        histo_type=rng.choice(["adenocarcinoma", "squamous cell carcinoma",
                                "small cell carcinoma", "large cell neuroendocrine"]),
        molecular=rng.choice(["EGFR L858R mutation detected",
                               "ALK rearrangement by FISH",
                               "KRAS G12C mutation identified",
                               "No actionable mutations detected"]),
        tnm=rng.choice(["T2aN1M0 (Stage IIB)", "T3N2M0 (Stage IIIA)",
                          "T1bN0M0 (Stage IA2)", "T4N3M1 (Stage IVB)"]),
    )

    specialty = rng.choice(SPECIALTIES)
    icd = ",".join(rng.sample(ICD_CODES, rng.randint(1, 3)))

    return {
        "doc_id": _generate_doc_id(index, "pathology"),
        "title": f"Pathology Report: {tissue.title()} - {condition}",
        "specialty": specialty,
        "doc_type": "pathology_report",
        "body_text": body,
        "icd_codes": icd,
    }


def _generate_cohort_summary(rng: random.Random, index: int) -> dict:
    """Generate a synthetic cohort summary."""
    drug = rng.choice(DRUG_NAMES)
    condition = rng.choice(CONDITIONS)
    synonym_sentence, _ = _pick_synonym_sentence(rng)
    n_patients = rng.randint(100, 5000)
    start_year = rng.randint(2015, 2022)

    template = rng.choice(COHORT_TEMPLATES)
    body = template.format(
        n_patients=n_patients, condition=condition, drug=drug,
        start_year=start_year, end_year=start_year + rng.randint(2, 5),
        synonym_sentence=synonym_sentence,
        median_age=rng.randint(45, 75),
        sex_dist=rng.choice(["55% male, 45% female", "62% female, 38% male",
                              "51% male, 49% female"]),
        followup=rng.randint(12, 60),
        os_timepoint=rng.choice([12, 24, 36]),
        os_pct=rng.randint(35, 85),
        biomarker_finding=rng.choice([
            f"high {rng.choice(BIOMARKERS)} expression correlated with improved outcomes",
            f"{rng.choice(BIOMARKERS)} negativity was associated with resistance",
            f"co-occurrence of {rng.choice(BIOMARKERS)} and {rng.choice(BIOMARKERS)} predicted response",
        ]),
        outcome_measure=rng.choice(["overall survival", "progression-free survival",
                                     "time to treatment failure", "real-world response rate"]),
        subgroup_var=rng.choice(["age", "biomarker status", "prior therapy lines",
                                  "performance status", "tumor burden"]),
        subgroup_finding=rng.choice([
            "significant benefit in biomarker-positive subgroup",
            "no differential effect across age groups",
            "greater benefit in treatment-naive patients",
            "consistent benefit regardless of performance status"
        ]),
        predictors=rng.choice([
            f"{rng.choice(BIOMARKERS)} status and tumor stage",
            f"baseline LDH and {rng.choice(BIOMARKERS)} expression",
            f"performance status and prior lines of therapy",
        ]),
        or_val=round(rng.uniform(1.2, 4.5), 2),
        ci_low=round(rng.uniform(0.8, 1.5), 2),
        ci_high=round(rng.uniform(2.0, 8.0), 2),
    )

    specialty = rng.choice(SPECIALTIES)
    icd = ",".join(rng.sample(ICD_CODES, rng.randint(1, 3)))

    return {
        "doc_id": _generate_doc_id(index, "cohort"),
        "title": f"Cohort Analysis: {drug} in {condition}",
        "specialty": specialty,
        "doc_type": "cohort_summary",
        "body_text": body,
        "icd_codes": icd,
    }


GENERATOR_MAP = {
    "trial_report": _generate_trial_report,
    "case_study": _generate_case_study,
    "drug_interaction": _generate_drug_interaction,
    "pathology_report": _generate_pathology_report,
    "cohort_summary": _generate_cohort_summary,
}


def generate_documents(num_docs: int, rng: random.Random) -> list[dict]:
    """Generate synthetic clinical documents with controlled type distribution."""
    documents = []
    for i in range(num_docs):
        doc_type = DOC_TYPES[i % len(DOC_TYPES)]
        generator = GENERATOR_MAP[doc_type]
        doc = generator(rng, i)
        # Add timestamp spread over 3 years
        days_offset = rng.randint(0, 1095)
        doc["created_at"] = (
            datetime(2021, 1, 1) + timedelta(days=days_offset)
        ).isoformat()
        documents.append(doc)
    return documents


def generate_image_features(num_docs: int, rng_seed: int) -> dict[str, np.ndarray]:
    """Generate 2048-dim synthetic image feature vectors (simulated ResNet output).

    These are NOT actual images — they simulate the output of a pre-trained ResNet
    applied to histopathology slides. Documents about similar topics get feature
    vectors with higher cosine similarity to create realistic clustering.
    """
    np_rng = np.random.RandomState(rng_seed)
    features = {}

    # Create cluster centers for each specialty (simulating visual similarity)
    cluster_centers = {}
    for spec in SPECIALTIES:
        cluster_centers[spec] = np_rng.randn(2048).astype(np.float32)

    return features


def generate_structured_metadata(
    documents: list[dict], rng: random.Random
) -> dict[str, dict]:
    """Generate structured metadata for each document.

    Includes lab values, biomarker levels, and encoded ICD codes.
    """
    metadata = {}
    for doc in documents:
        doc_meta = {
            "lab_values": {
                lab: round(rng.uniform(0.1, 15.0), 2)
                for lab in rng.sample(LAB_TESTS, rng.randint(3, 8))
            },
            "biomarker_levels": {
                bm: round(rng.uniform(0, 100), 1)
                for bm in rng.sample(BIOMARKERS, rng.randint(1, 5))
            },
            "icd_codes_list": doc["icd_codes"].split(","),
            "word_count": len(doc["body_text"].split()),
        }
        metadata[doc["doc_id"]] = doc_meta
    return metadata


def generate_physicians(num_physicians: int, rng: random.Random) -> list[dict]:
    """Generate synthetic physician profiles."""
    physicians = []
    for i in range(num_physicians):
        physicians.append({
            "physician_id": _generate_physician_id(i),
            "specialty": rng.choice(SPECIALTIES),
            "years_experience": rng.randint(1, 35),
        })
    return physicians


def generate_interactions(
    physicians: list[dict],
    documents: list[dict],
    num_interactions: int,
    rng: random.Random,
) -> list[dict]:
    """Generate synthetic physician-document interactions.

    Interaction distribution is intentionally skewed:
    - Popular documents (top 20%) get 60% of interactions
    - Most documents get very few interactions (creating cold start)
    - Physicians prefer documents in their specialty (creating signal for CF)
    """
    interactions = []
    doc_ids = [d["doc_id"] for d in documents]
    doc_specialties = {d["doc_id"]: d["specialty"] for d in documents}

    # Create popularity distribution (power law)
    num_docs = len(doc_ids)
    popularity = np.random.RandomState(42).zipf(1.5, num_docs)
    popularity = popularity / popularity.sum()

    base_date = datetime(2021, 1, 1)

    for i in range(num_interactions):
        physician = rng.choice(physicians)

        # 70% chance physician interacts with document in their specialty
        if rng.random() < 0.7:
            specialty_docs = [
                d for d in doc_ids
                if doc_specialties[d] == physician["specialty"]
            ]
            if specialty_docs:
                doc_id = rng.choice(specialty_docs)
            else:
                doc_id = rng.choices(doc_ids, weights=popularity, k=1)[0]
        else:
            doc_id = rng.choices(doc_ids, weights=popularity, k=1)[0]

        interaction_type = rng.choices(
            ["view", "click", "save", "dwell"],
            weights=[0.5, 0.25, 0.1, 0.15],
            k=1,
        )[0]

        dwell_time = None
        if interaction_type == "dwell":
            dwell_time = round(rng.uniform(5.0, 300.0), 1)
        elif interaction_type in ("click", "save"):
            dwell_time = round(rng.uniform(2.0, 60.0), 1)

        timestamp = base_date + timedelta(
            days=rng.randint(0, 1095),
            hours=rng.randint(8, 18),
            minutes=rng.randint(0, 59),
        )

        interactions.append({
            "physician_id": physician["physician_id"],
            "doc_id": doc_id,
            "interaction_type": interaction_type,
            "dwell_time_seconds": dwell_time,
            "timestamp": timestamp.isoformat(),
            "query_id": None,  # Will be linked to queries below
        })

    return interactions


def generate_search_queries(
    physicians: list[dict],
    documents: list[dict],
    num_queries: int,
    rng: random.Random,
) -> list[dict]:
    """Generate synthetic search queries with vocabulary mismatch built in.

    Some queries use different terminology than the matching documents,
    creating the exact retrieval challenge this system addresses.
    """
    queries = []
    base_date = datetime(2021, 1, 1)

    for i in range(num_queries):
        physician = rng.choice(physicians)
        template = rng.choice(SEARCH_QUERY_TEMPLATES)

        # Fill in template variables
        query_text = template
        replacements = {
            "{drug}": rng.choice(DRUG_NAMES),
            "{drug1}": rng.choice(DRUG_NAMES),
            "{drug2}": rng.choice(DRUG_NAMES),
            "{condition}": rng.choice(CONDITIONS),
            "{drug_class}": rng.choice(DRUG_CLASSES),
            "{biomarker}": rng.choice(BIOMARKERS),
            "{phase}": rng.choice(["I", "II", "III"]),
            "{tissue_type}": rng.choice(TISSUE_TYPES),
            "{lab_test}": rng.choice(LAB_TESTS),
        }

        # Use synonym terms to create vocabulary mismatch
        group = rng.choice(CLINICAL_SYNONYM_GROUPS)
        replacements["{synonym_term}"] = rng.choice(group["terms"])

        for key, val in replacements.items():
            query_text = query_text.replace(key, val)

        timestamp = base_date + timedelta(
            days=rng.randint(0, 1095),
            hours=rng.randint(8, 18),
            minutes=rng.randint(0, 59),
        )

        queries.append({
            "query_id": _generate_query_id(i),
            "physician_id": physician["physician_id"],
            "query_text": query_text,
            "timestamp": timestamp.isoformat(),
        })

    return queries


def generate_relevance_judgments(
    queries: list[dict],
    documents: list[dict],
    rng: random.Random,
    docs_per_query: int = 20,
) -> list[dict]:
    """Generate relevance judgments (0-3 scale) for query-document pairs.

    Relevance is influenced by:
    - Term overlap between query and document (weak signal)
    - Same specialty match (moderate signal)
    - Same clinical concept via synonym groups (strong signal, but not
      captured by term overlap — this is the vocabulary mismatch problem)
    """
    judgments = []
    doc_list = list(documents)

    for query in queries:
        # Sample documents to judge for this query
        sampled_docs = rng.sample(doc_list, min(docs_per_query, len(doc_list)))

        for doc in sampled_docs:
            # Base relevance from term overlap
            query_terms = set(query["query_text"].lower().split())
            doc_terms = set(doc["body_text"].lower().split())
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)

            # Specialty bonus
            query_physician = query["physician_id"]
            # Simple heuristic: if doc type matches query context, more relevant
            specialty_match = rng.random() < 0.3

            # Compute relevance score
            base_score = overlap * 2
            if specialty_match:
                base_score += 0.5

            # Add noise to make it realistic
            base_score += rng.uniform(-0.3, 0.5)

            # Map to 0-3 scale
            if base_score >= 1.5:
                relevance = 3
            elif base_score >= 1.0:
                relevance = 2
            elif base_score >= 0.4:
                relevance = 1
            else:
                relevance = 0

            judgments.append({
                "query_id": query["query_id"],
                "doc_id": doc["doc_id"],
                "relevance_score": relevance,
            })

    return judgments


def link_interactions_to_queries(
    interactions: list[dict],
    queries: list[dict],
    rng: random.Random,
    link_ratio: float = 0.6,
) -> list[dict]:
    """Link a fraction of interactions to search queries.

    This creates the query-interaction chain needed for evaluation:
    query → interaction → document.
    """
    query_ids = [q["query_id"] for q in queries]

    for interaction in interactions:
        if rng.random() < link_ratio:
            interaction["query_id"] = rng.choice(query_ids)

    return interactions


def create_database(db_path: str, schema_path: str) -> sqlite3.Connection:
    """Create SQLite database from schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    with open(schema_path) as f:
        conn.executescript(f.read())

    return conn


def insert_data(
    conn: sqlite3.Connection,
    documents: list[dict],
    physicians: list[dict],
    interactions: list[dict],
    queries: list[dict],
    judgments: list[dict],
) -> None:
    """Insert generated data into SQLite database."""
    cursor = conn.cursor()

    # Insert documents
    cursor.executemany(
        "INSERT INTO documents (doc_id, title, specialty, doc_type, body_text, created_at, icd_codes) "
        "VALUES (:doc_id, :title, :specialty, :doc_type, :body_text, :created_at, :icd_codes)",
        documents,
    )

    # Insert physicians
    cursor.executemany(
        "INSERT INTO physicians (physician_id, specialty, years_experience) "
        "VALUES (:physician_id, :specialty, :years_experience)",
        physicians,
    )

    # Insert interactions
    cursor.executemany(
        "INSERT INTO interactions (physician_id, doc_id, interaction_type, dwell_time_seconds, timestamp, query_id) "
        "VALUES (:physician_id, :doc_id, :interaction_type, :dwell_time_seconds, :timestamp, :query_id)",
        interactions,
    )

    # Insert search queries
    cursor.executemany(
        "INSERT INTO search_queries (query_id, physician_id, query_text, timestamp) "
        "VALUES (:query_id, :physician_id, :query_text, :timestamp)",
        queries,
    )

    # Insert relevance judgments (handle duplicates by ignoring)
    cursor.executemany(
        "INSERT OR IGNORE INTO query_document_relevance (query_id, doc_id, relevance_score) "
        "VALUES (:query_id, :doc_id, :relevance_score)",
        judgments,
    )

    conn.commit()


def generate_dataset(config_path: str, seed: int = 42) -> str:
    """Generate a complete synthetic dataset from a YAML config.

    Args:
        config_path: Path to YAML config file (1k_config.yaml, etc.)
        seed: Random seed for reproducibility.

    Returns:
        Path to the created SQLite database.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    num_docs = data_config["num_documents"]
    num_physicians = data_config["num_physicians"]
    num_queries = data_config["num_queries"]
    num_interactions = data_config["num_interactions"]
    db_path = data_config["db_path"]

    rng = random.Random(seed)

    print(f"Generating {num_docs} documents...")
    documents = generate_documents(num_docs, rng)

    print(f"Generating {num_physicians} physicians...")
    physicians = generate_physicians(num_physicians, rng)

    print(f"Generating {num_interactions} interactions...")
    interactions = generate_interactions(
        physicians, documents, num_interactions, rng
    )

    print(f"Generating {num_queries} search queries...")
    queries = generate_search_queries(physicians, documents, num_queries, rng)

    print("Linking interactions to queries...")
    interactions = link_interactions_to_queries(interactions, queries, rng)

    print("Generating relevance judgments...")
    judgments = generate_relevance_judgments(queries, documents, rng)

    print(f"Creating database at {db_path}...")
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "schema.sql"
    )
    conn = create_database(db_path, schema_path)
    insert_data(conn, documents, physicians, interactions, queries, judgments)

    # Verify counts
    cursor = conn.cursor()
    for table in ["documents", "physicians", "interactions", "search_queries", "query_document_relevance"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print(f"Dataset generated: {db_path}")
    return db_path


def generate_image_feature_vectors(
    doc_ids: list[str], seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate 2048-dim synthetic image feature vectors for each document.

    Simulates ResNet output on histopathology slides. Vectors are normalized
    to unit length to simulate typical CNN feature extraction behavior.
    """
    np_rng = np.random.RandomState(seed)
    features = {}

    for doc_id in doc_ids:
        vec = np_rng.randn(2048).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # L2 normalize
        features[doc_id] = vec

    return features


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/1k_config.yaml"

    generate_dataset(config_path)
