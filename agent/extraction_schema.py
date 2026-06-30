"""
Structured extraction schema for clinical trial protocols.

Mirrors a standard structured-extraction pattern: take unstructured/
semi-structured text and turn it into a validated, machine-readable
record (via a Pydantic schema with an extract -> validate -> auto-repair
loop) that an agent or retrieval system can reliably query.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class TreatmentArm(BaseModel):
    arm_label: str = Field(description="e.g. 'Arm 1', 'Arm 2'")
    description: str = Field(description="e.g. 'Gabapentin Low Dose'")


class EligibilityCriteria(BaseModel):
    inclusion: List[str] = Field(default_factory=list)
    exclusion: List[str] = Field(default_factory=list)


class EfficacyEndpoints(BaseModel):
    primary: List[str] = Field(default_factory=list)
    secondary: List[str] = Field(default_factory=list)


class StudyProtocol(BaseModel):
    """Top-level structured representation of a single trial protocol."""

    protocol_number: str
    title: str
    phase: str
    study_design: str
    therapeutic_area: Optional[str] = None
    indication: Optional[str] = Field(
        default=None, description="Disease/condition under study, e.g. 'Leukemia'"
    )
    treatment_duration_weeks: Optional[int] = None
    total_study_duration_weeks: Optional[int] = None

    primary_objective: Optional[str] = None
    secondary_objectives: List[str] = Field(default_factory=list)

    patient_population: Optional[str] = None
    planned_enrollment: Optional[int] = None
    number_of_sites: Optional[int] = None
    treatment_arms: List[TreatmentArm] = Field(default_factory=list)

    eligibility: EligibilityCriteria = Field(default_factory=EligibilityCriteria)
    endpoints: EfficacyEndpoints = Field(default_factory=EfficacyEndpoints)

    sample_size_rationale: Optional[str] = None
    statistical_methods: Optional[str] = None

    source_file: Optional[str] = Field(
        default=None, description="Filename this record was extracted from, for traceability"
    )
