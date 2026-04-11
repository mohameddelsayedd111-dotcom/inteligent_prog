"""
rule_based_system/rules.py
Defines at least 10 Experta-compatible rules for heart disease risk.
"""

# Python 3.12 compatibility patch for Experta
import collections
import collections.abc
for _attr in ('Mapping', 'MutableMapping', 'Callable', 'Iterable',
              'Iterator', 'Sequence', 'MutableSequence', 'Set', 'MutableSet'):
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

from experta import Fact, KnowledgeEngine, Rule, L, P, AS, MATCH


class PatientData(Fact):
    """
    Fact schema holding patient health indicators.
    Fields:
        age (int), sex (int 0/1), chol (float mg/dl),
        trestbps (float mmHg), thalach (float bpm),
        fbs (int 0/1), exang (int 0/1),
        cp (int 0-3), oldpeak (float), ca (int 0-3)
    """
    pass


class HeartDiseaseExpertSystem(KnowledgeEngine):
    """
    Expert system with ≥10 rules for heart disease risk assessment.
    Fires rules and accumulates a risk score.
    """

    def __init__(self):
        super().__init__()
        self.risk_score = 0
        self.fired_rules = []

    # ─────────────────────────────────────────────
    # HIGH-RISK RULES
    # ─────────────────────────────────────────────

    @Rule(PatientData(chol=P(lambda c: c > 240), age=P(lambda a: a > 50)))
    def rule_01_high_chol_elderly(self):
        """Rule 1: High cholesterol + age > 50 → High risk."""
        self.risk_score += 30
        self.fired_rules.append("R01: High cholesterol (>240) & age >50 → +30 risk")

    @Rule(PatientData(trestbps=P(lambda bp: bp > 140), exang=L(1)))
    def rule_02_hypertension_exang(self):
        """Rule 2: Hypertension + exercise-induced angina → High risk."""
        self.risk_score += 35
        self.fired_rules.append("R02: Hypertension (>140 mmHg) & exercise angina → +35 risk")

    @Rule(PatientData(cp=P(lambda cp: cp in [0, 1])))
    def rule_03_typical_chest_pain(self):
        """Rule 3: Typical / atypical chest pain → elevated risk."""
        self.risk_score += 25
        self.fired_rules.append("R03: Chest pain type (typical/atypical) → +25 risk")

    @Rule(PatientData(ca=P(lambda ca: ca >= 2)))
    def rule_04_major_vessels(self):
        """Rule 4: ≥2 major vessels blocked → High risk."""
        self.risk_score += 35
        self.fired_rules.append("R04: ≥2 major vessels colored by fluoroscopy → +35 risk")

    @Rule(PatientData(oldpeak=P(lambda op: op > 2.0)))
    def rule_05_st_depression(self):
        """Rule 5: ST depression > 2.0 → High risk."""
        self.risk_score += 30
        self.fired_rules.append("R05: ST depression >2.0 → +30 risk")

    @Rule(PatientData(fbs=L(1), age=P(lambda a: a > 45)))
    def rule_06_diabetes_age(self):
        """Rule 6: Fasting blood sugar > 120 mg/dl + age > 45 → elevated risk."""
        self.risk_score += 20
        self.fired_rules.append("R06: Fasting blood sugar elevated & age >45 → +20 risk")

    @Rule(PatientData(thalach=P(lambda th: th < 120)))
    def rule_07_low_max_hr(self):
        """Rule 7: Max heart rate < 120 bpm → concerning."""
        self.risk_score += 20
        self.fired_rules.append("R07: Low maximum heart rate (<120 bpm) → +20 risk")

    @Rule(PatientData(sex=L(1), age=P(lambda a: a > 55),
                      chol=P(lambda c: c > 200)))
    def rule_08_male_age_chol(self):
        """Rule 8: Male + age >55 + cholesterol >200 → compound risk."""
        self.risk_score += 25
        self.fired_rules.append("R08: Male, age >55, chol >200 → +25 risk")

    @Rule(PatientData(exang=L(1), oldpeak=P(lambda op: op > 1.5)))
    def rule_09_angina_st(self):
        """Rule 9: Exercise angina + ST depression → compound risk."""
        self.risk_score += 30
        self.fired_rules.append("R09: Exercise angina + ST depression >1.5 → +30 risk")

    @Rule(PatientData(trestbps=P(lambda bp: bp > 160)))
    def rule_10_severe_hypertension(self):
        """Rule 10: Severe hypertension (>160 mmHg) → High risk."""
        self.risk_score += 35
        self.fired_rules.append("R10: Severe hypertension (>160 mmHg) → +35 risk")

    # ─────────────────────────────────────────────
    # LOW-RISK / PROTECTIVE RULES
    # ─────────────────────────────────────────────

    @Rule(PatientData(thalach=P(lambda th: th >= 150),
                      chol=P(lambda c: c <= 200)))
    def rule_11_healthy_hr_chol(self):
        """Rule 11: High max HR + normal cholesterol → protective."""
        self.risk_score -= 15
        self.fired_rules.append("R11: Good max HR (≥150) & normal cholesterol → -15 risk")

    @Rule(PatientData(exang=L(0), oldpeak=P(lambda op: op <= 1.0),
                      ca=L(0)))
    def rule_12_low_risk_profile(self):
        """Rule 12: No exercise angina + low ST + 0 blocked vessels → Low risk."""
        self.risk_score -= 20
        self.fired_rules.append("R12: No angina, low ST depression, 0 vessels → -20 risk")

    def get_risk_level(self) -> str:
        if self.risk_score >= 60:
            return "HIGH"
        elif self.risk_score >= 25:
            return "MODERATE"
        else:
            return "LOW"
