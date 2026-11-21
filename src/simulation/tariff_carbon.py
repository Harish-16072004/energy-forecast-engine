from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TariffPlan:
    name: str
    off_peak_rate_per_kwh: float
    peak_rate_per_kwh: float
    peak_hours: tuple[int, ...]
    demand_charge_per_kw: float
    fixed_monthly_fee: float = 0.0


DEFAULT_TARIFF_PLANS: list[TariffPlan] = [
    TariffPlan(
        name="Commercial Flat Rate",
        off_peak_rate_per_kwh=0.14,
        peak_rate_per_kwh=0.14,
        peak_hours=(),
        demand_charge_per_kw=5.0,
        fixed_monthly_fee=25.0,
    ),
    TariffPlan(
        name="Commercial Time-of-Use (ToU)",
        off_peak_rate_per_kwh=0.09,
        peak_rate_per_kwh=0.24,
        peak_hours=(13, 14, 15, 16, 17, 18, 19, 20),  # Afternoon/evening peak hours (UTC or local)
        demand_charge_per_kw=12.0,
        fixed_monthly_fee=35.0,
    ),
]


def calculate_plan_cost(
    hourly_kwh: Iterable[float],
    hours_of_day: Iterable[int],
    plan: TariffPlan,
    grid_kg_co2_per_kwh: float = 0.40,
) -> dict:
    kwh_list = [max(0.0, float(x)) for x in hourly_kwh]
    hour_list = [int(h) for h in hours_of_day]

    if len(kwh_list) != len(hour_list):
        raise ValueError("hourly_kwh and hours_of_day must have the same length.")
    if not kwh_list:
        raise ValueError("hourly_kwh cannot be empty.")

    peak_set = set(plan.peak_hours)
    energy_cost = 0.0
    peak_kwh = 0.0
    off_peak_kwh = 0.0

    for val, hr in zip(kwh_list, hour_list):
        if hr in peak_set:
            energy_cost += val * plan.peak_rate_per_kwh
            peak_kwh += val
        else:
            energy_cost += val * plan.off_peak_rate_per_kwh
            off_peak_kwh += val

    # For hourly intervals (1 hour duration), average kWh in an hour equals average kW over that hour
    peak_demand_kw = max(kwh_list)
    demand_charge = peak_demand_kw * plan.demand_charge_per_kw
    total_cost = energy_cost + demand_charge + plan.fixed_monthly_fee
    total_kwh = sum(kwh_list)
    scope2_kg_co2 = total_kwh * grid_kg_co2_per_kwh

    return {
        "plan_name": plan.name,
        "total_kwh": round(total_kwh, 2),
        "peak_kwh": round(peak_kwh, 2),
        "off_peak_kwh": round(off_peak_kwh, 2),
        "peak_demand_kw": round(peak_demand_kw, 2),
        "energy_cost_usd": round(energy_cost, 2),
        "demand_charge_usd": round(demand_charge, 2),
        "fixed_fee_usd": round(plan.fixed_monthly_fee, 2),
        "total_cost_usd": round(total_cost, 2),
        "scope2_emissions_kg_co2": round(scope2_kg_co2, 2),
    }


def recommend_best_tariff(
    hourly_kwh: Iterable[float],
    hours_of_day: Iterable[int],
    plans: list[TariffPlan] | None = None,
    grid_kg_co2_per_kwh: float = 0.40,
) -> dict:
    candidate_plans = plans or DEFAULT_TARIFF_PLANS
    results = [
        calculate_plan_cost(hourly_kwh, hours_of_day, plan, grid_kg_co2_per_kwh)
        for plan in candidate_plans
    ]
    results_sorted = sorted(results, key=lambda r: r["total_cost_usd"])
    best = results_sorted[0]
    worst = results_sorted[-1]

    savings_vs_highest_usd = round(worst["total_cost_usd"] - best["total_cost_usd"], 2)
    savings_pct = (
        round((savings_vs_highest_usd / worst["total_cost_usd"]) * 100, 2)
        if worst["total_cost_usd"] > 0
        else 0.0
    )

    return {
        "recommended_plan": best["plan_name"],
        "estimated_savings_vs_highest_usd": savings_vs_highest_usd,
        "estimated_savings_pct": savings_pct,
        "all_plans": results_sorted,
    }