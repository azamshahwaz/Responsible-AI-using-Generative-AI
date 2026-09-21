def calculate_governance_score(
    edqs,
    fairness,
    accuracy,
    eri
):

    governance_score = (

        0.30 * edqs

        +

        0.25 * (
            fairness * 100
        )

        +

        0.20 * (
            accuracy * 100
        )

        +

        0.25 * (
            100 -
            (eri * 100)
        )

    )

    print(
        "\n========== GOVERNANCE SCORE =========="
    )

    print(
        f"\nGovernance Score : "
        f"{governance_score:.2f}"
    )

    if governance_score >= 85:

        level = "EXCELLENT"

    elif governance_score >= 70:

        level = "GOOD"

    elif governance_score >= 50:

        level = "MODERATE"

    else:

        level = "POOR"

    print(
        f"Governance Level : "
        f"{level}"
    )

    return {
        "score": governance_score,
        "level": level
    }