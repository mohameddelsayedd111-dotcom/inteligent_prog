"""
rule_based_system/expert_system.py
Inference engine for the heart disease expert system.
"""

import collections
import collections.abc
for _a in ("Mapping","MutableMapping","Callable","Iterable","Iterator","Sequence"):
    if not hasattr(collections, _a): setattr(collections, _a, getattr(collections.abc, _a))
from rules import HeartDiseaseExpertSystem, PatientData


def run_expert_system(patient: dict) -> dict:
    """
    Run the expert system on a patient dict and return results.

    patient keys:
        age, sex, chol, trestbps, thalach,
        fbs, exang, cp, oldpeak, ca
    """
    engine = HeartDiseaseExpertSystem()
    engine.reset()
    engine.declare(PatientData(**patient))
    engine.run()

    risk_level = engine.get_risk_level()
    return {
        "risk_score": engine.risk_score,
        "risk_level": risk_level,
        "fired_rules": engine.fired_rules,
    }


def interactive_session():
    """Allow user to input patient data and receive a risk prediction."""
    print("=" * 55)
    print("  HEART DISEASE EXPERT SYSTEM — Patient Intake")
    print("=" * 55)

    def ask(prompt, cast=int, lo=None, hi=None):
        while True:
            try:
                val = cast(input(f"  {prompt}: "))
                if lo is not None and val < lo:
                    print(f"    ⚠  Must be ≥ {lo}")
                    continue
                if hi is not None and val > hi:
                    print(f"    ⚠  Must be ≤ {hi}")
                    continue
                return val
            except ValueError:
                print("    ⚠  Invalid input, please try again.")

    patient = {
        "age":      ask("Age (years)", int, 1, 120),
        "sex":      ask("Sex [0=Female, 1=Male]", int, 0, 1),
        "chol":     ask("Cholesterol (mg/dl)", float, 50, 600),
        "trestbps": ask("Resting Blood Pressure (mmHg)", float, 60, 250),
        "thalach":  ask("Maximum Heart Rate Achieved (bpm)", float, 50, 250),
        "fbs":      ask("Fasting Blood Sugar > 120 mg/dl [0=No, 1=Yes]", int, 0, 1),
        "exang":    ask("Exercise Induced Angina [0=No, 1=Yes]", int, 0, 1),
        "cp":       ask("Chest Pain Type [0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic]", int, 0, 3),
        "oldpeak":  ask("ST Depression (oldpeak)", float, 0, 10),
        "ca":       ask("Number of Major Vessels Colored (0-3)", int, 0, 3),
    }

    result = run_expert_system(patient)

    print("\n" + "=" * 55)
    print(f"  RISK SCORE  : {result['risk_score']}")
    print(f"  RISK LEVEL  : {result['risk_level']}")
    print("\n  Rules Fired:")
    for r in result["fired_rules"]:
        print(f"    ✦ {r}")
    print("=" * 55)
    return result


if __name__ == "__main__":
    interactive_session()
