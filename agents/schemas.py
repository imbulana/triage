from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from cfpb_taxonomy import cfpb_label_sets


CFPB_LABEL_SETS = cfpb_label_sets(include_unknown=True)


ProductLabel = Literal[
    "banking",
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "debt_or_credit_management",
    "money_transfer",
    "mortgage",
    "payday_personal_loan",
    "prepaid_card",
    "student_loan",
    "vehicle_loan",
    "unknown",
]

IssueLabel = Literal[
    "account_access",
    "account_opening_or_closure",
    "billing_or_payment_error",
    "discrimination_or_udaap",
    "dispute_handling",
    "fee_or_penalty",
    "funds_availability",
    "servicing_delay",
    "transaction_or_transfer_error",
    "unauthorized_transaction",
    "unknown",
]

RouteLabel = Literal[
    "card_ops",
    "credit_reporting_ops",
    "digital_banking",
    "lending_ops",
    "triage_ops",
]


class DomainClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: ProductLabel = Field(description="Internal normalized product label.")
    cfpb_product: str = Field(
        description="Exact CFPB Product taxonomy string, e.g. 'Checking or savings account'. Do not normalize.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["products"])},
    )
    cfpb_sub_product: str = Field(
        description="Exact CFPB Sub-product taxonomy string, e.g. 'Checking account'. Do not normalize.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["sub_products"])},
    )
    issue: IssueLabel = Field(description="Internal normalized issue label.")
    cfpb_issue: str = Field(
        description="Exact CFPB Issue taxonomy string, e.g. 'Managing an account'. Do not normalize.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["issues"])},
    )
    sub_issue: Optional[str] = Field(default=None, description="Internal normalized sub-issue label or null.")
    cfpb_sub_issue: str = Field(
        description="Exact CFPB Sub-issue taxonomy string, e.g. 'Funds not handled or disbursed as instructed'. Do not normalize.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["sub_issues"])},
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class CFPBDomainClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(
        description="Exact CFPB Product taxonomy string, e.g. 'Checking or savings account'.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["products"])},
    )
    sub_product: str = Field(
        description="Exact CFPB Sub-product taxonomy string, e.g. 'Checking account'.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["sub_products"])},
    )
    issue: str = Field(
        description="Exact CFPB Issue taxonomy string, e.g. 'Managing an account'.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["issues"])},
    )
    sub_issue: str = Field(
        description="Exact CFPB Sub-issue taxonomy string, e.g. 'Funds not handled or disbursed as instructed'.",
        json_schema_extra={"enum": list(CFPB_LABEL_SETS["sub_issues"])},
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ComplianceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compliance_risk: float = Field(ge=0.0, le=1.0)
    veto: bool
    policy_checks: List[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: RouteLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ResolutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_team: RouteLabel
    actions: List[str] = Field(min_length=1, max_length=8)
    customer_response: str
    preventive_recommendations: List[str] = Field(default_factory=list, max_length=6)


class ResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution_plan: ResolutionPlan
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
