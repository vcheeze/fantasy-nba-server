import random
from collections import Counter
from itertools import combinations

import pytest
from ortools.sat.python import cp_model

from main import select_player_pool
from solver import OptimizationConfig, TeamOptimizer, Transfers

N_TEAMS = 10
PHASES = [
    {"id": 1, "start_event": 1, "stop_event": 999},
    {"id": 2, "start_event": 1, "stop_event": 2},
    {"id": 3, "start_event": 3, "stop_event": 999},
]
TEAMS = [{"id": t, "name": f"Team {t}", "short_name": f"T{t}"} for t in range(1, N_TEAMS + 1)]
CONFIG = OptimizationConfig(SOLVER_TIMEOUT=60, RELATIVE_GAP_LIMIT=0, STALL_SECONDS=None)


def player(pid, element_type, team, score, cost, status="a", minutes=100, chance=None):
    return {
        "id": pid, "web_name": f"P{pid}", "element_type": element_type, "team": team,
        "form": str(score), "points_per_game": str(score), "now_cost": cost,
        "status": status, "minutes": minutes, "chance_of_playing_next_round": chance,
    }


def make_league(seed, n_players=50, n_events=4):
    rng = random.Random(seed)
    players = []
    for pid in range(1, n_players + 1):
        score = round(rng.uniform(150, 550), 1)
        cost = int(30 + score / 550 * 130 + rng.randint(-15, 15))
        players.append(player(pid, 1 if pid % 2 else 2, rng.randint(1, N_TEAMS), score, cost))
    fixtures = []
    for event in range(1, n_events + 1):
        teams = list(range(1, N_TEAMS + 1))
        rng.shuffle(teams)
        for k in range(rng.randint(2, 5)):
            fixtures.append({"event": event, "team_h": teams[2 * k], "team_a": teams[2 * k + 1]})
    return players, fixtures


def legal_squad(players):
    """Cheapest legal 5+5 squad respecting the team limit."""
    squad, per_team = [], Counter()
    for element_type in (1, 2):
        count = 0
        for p in sorted((p for p in players if p["element_type"] == element_type), key=lambda p: p["now_cost"]):
            if count < 5 and per_team[p["team"]] < 2:
                squad.append(p)
                per_team[p["team"]] += 1
                count += 1
    assert len(squad) == 10
    return [{"element": p["id"], "selling_price": p["now_cost"]} for p in squad]


def solve(players, fixtures, events, squad=None, transfers=None, **kwargs):
    optimizer = TeamOptimizer(players, fixtures, PHASES, TEAMS, "form", squad, CONFIG, transfers, **kwargs)
    return optimizer.optimize(events)


def reference_optimum(players, fixtures, events, squad=None, bank=0, free_first=2):
    """Independent model of the official rules; returns net points (score minus hits)."""
    model = cp_model.CpModel()
    by_id = {p["id"]: p for p in players}
    selling = {s["element"]: s["selling_price"] for s in squad or []}
    cost = {i: selling.get(i, p["now_cost"]) for i, p in by_id.items()}
    score = {i: round(float(p["form"]) * 10) for i, p in by_id.items()}
    budget = sum(selling.values()) + bank if squad else 1000
    playing = {e: {t for f in fixtures if f["event"] == e for t in (f["team_h"], f["team_a"])} for e in events}
    gameweek = {e: next(ph["id"] for ph in PHASES[1:] if ph["start_event"] <= e <= ph["stop_event"]) for e in events}

    x = {(i, e): model.NewBoolVar("") for i in by_id for e in events}
    start = {(i, e): model.NewBoolVar("") for i in by_id for e in events}
    joined = {(i, e): model.NewBoolVar("") for i in by_id for e in events}
    objective = []
    for k, e in enumerate(events):
        for element_type in (1, 2):
            model.Add(sum(x[i, e] for i in by_id if by_id[i]["element_type"] == element_type) == 5)
            model.Add(sum(start[i, e] for i in by_id if by_id[i]["element_type"] == element_type) <= 3)
        model.Add(sum(start[i, e] for i in by_id) <= 5)
        for team in range(1, N_TEAMS + 1):
            model.Add(sum(x[i, e] for i in by_id if by_id[i]["team"] == team) <= 2)
        model.Add(sum(x[i, e] * cost[i] for i in by_id) <= budget)
        for i in by_id:
            model.Add(start[i, e] <= x[i, e])
            if by_id[i]["team"] not in playing[e]:
                model.Add(start[i, e] == 0)
            if k == 0 and not squad:
                model.Add(joined[i, e] == 0)
            else:
                before = (1 if i in selling else 0) if k == 0 else x[i, events[k - 1]]
                model.Add(joined[i, e] >= x[i, e] - before)
            objective.append(start[i, e] * score[i])
    for gw in set(gameweek.values()):
        total = sum(joined[i, e] for i in by_id for e in events if gameweek[e] == gw)
        excess = model.NewIntVar(0, 1000, "")
        model.Add(excess >= total - (free_first if gw == gameweek[events[0]] else 2))
        objective.append(-excess * 10000)
    model.Maximize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_workers = 8
    assert solver.Solve(model) == cp_model.OPTIMAL
    return solver.ObjectiveValue() / 10


def best_lineup_score(roster_ids, players, playing_teams):
    """Brute force over every legal 5-man (or smaller) line-up."""
    by_id = {p["id"]: p for p in players}
    playing = [by_id[i] for i in roster_ids if by_id[i]["team"] in playing_teams]
    best = 0.0
    for size in range(min(5, len(playing)) + 1):
        for combo in combinations(playing, size):
            counts = Counter(p["element_type"] for p in combo)
            if counts[1] <= 3 and counts[2] <= 3:
                best = max(best, sum(float(p["form"]) for p in combo))
    return best


def net_score(result):
    return result["true_gameweek_score"] - result["transfers"]["cost"]


def assert_legal(result, players, fixtures, budget):
    by_id = {p["id"]: p for p in players}
    for event, roster in result["squad_by_event"].items():
        ids = [p["id"] for p in roster]
        assert len(ids) == 10
        assert Counter(by_id[i]["element_type"] for i in ids) == {1: 5, 2: 5}
        assert max(Counter(by_id[i]["team"] for i in ids).values()) <= 2
        assert sum(p["cost"] for p in roster) <= budget
        playing = {t for f in fixtures if f["event"] == int(event) for t in (f["team_h"], f["team_a"])}
        assert result["daily_starters"][int(event)]["score"] == pytest.approx(
            best_lineup_score(ids, players, playing)
        )


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference_when_building_from_scratch(seed):
    players, fixtures = make_league(seed)
    result = solve(players, fixtures, [1, 2, 3, 4])
    assert result["status"] == "Optimal"
    assert_legal(result, players, fixtures, 1000)
    assert net_score(result) == pytest.approx(reference_optimum(players, fixtures, [1, 2, 3, 4]))


@pytest.mark.parametrize("seed", range(6))
def test_matches_reference_with_existing_squad(seed):
    players, fixtures = make_league(seed)
    squad = legal_squad(players)
    bank = 150
    transfers = Transfers(bank=bank, cost=1000, limit=2, made=1, status="cost", value=0)
    result = solve(players, fixtures, [1, 2, 3, 4], squad, transfers)
    assert result["status"] == "Optimal"
    budget = sum(s["selling_price"] for s in squad) + bank
    assert_legal(result, players, fixtures, budget)
    assert net_score(result) == pytest.approx(
        reference_optimum(players, fixtures, [1, 2, 3, 4], squad, bank, free_first=1)
    )


def test_identical_players_on_different_teams_are_not_interchangeable():
    players = [player(10 * t + et, et, t, 100, 50) for t in range(1, 6) for et in (1, 2)]
    players += [player(100, 2, 9, 500, 500), player(101, 2, 10, 500, 500)]
    fixtures = [{"event": e, "team_h": 10, "team_a": 1} for e in (1, 2)]
    result = solve(players, fixtures, [1, 2])
    assert 101 in {p["id"] for p in result["squad"]}
    assert result["true_gameweek_score"] == pytest.approx(1400)


def test_injured_squad_player_is_replaced():
    players, fixtures = make_league(3)
    squad = legal_squad(players)
    injured_id = squad[0]["element"]
    for p in players:
        if p["id"] == injured_id:
            p.update(status="i", chance_of_playing_next_round=0, form="550")
    transfers = Transfers(bank=300, cost=1000, limit=2, made=0, status="cost", value=0)
    result = solve(players, fixtures, [1, 2], squad, transfers)
    assert injured_id not in {p["id"] for p in result["squad"]}


def test_zero_minute_squad_player_stays_in_pool():
    players, fixtures = make_league(1)
    squad = legal_squad(players)
    players[squad[0]["element"] - 1]["minutes"] = 0
    pool = select_player_pool(players, squad)
    assert squad[0]["element"] in {p["id"] for p in pool}
    transfers = Transfers(bank=300, cost=1000, limit=2, made=0, status="cost", value=0)
    assert solve(pool, fixtures, [1, 2], squad, transfers)["status"] == "Optimal"


def test_non_contiguous_gamedays():
    players, fixtures = make_league(2)
    remap = {1: 1, 2: 3, 3: 5, 4: 7}
    fixtures = [dict(f, event=remap[f["event"]]) for f in fixtures]
    squad = legal_squad(players)
    transfers = Transfers(bank=300, cost=1000, limit=2, made=0, status="cost", value=0)
    result = solve(players, fixtures, [1, 3, 5, 7], squad, transfers)
    assert result["status"] == "Optimal"
    assert_legal(result, players, fixtures, sum(s["selling_price"] for s in squad) + 300)


def test_unlimited_transfers_status_makes_first_gameday_free():
    players, fixtures = make_league(4)
    squad = legal_squad(players)
    transfers = Transfers(bank=300, cost=0, limit=None, made=0, status="unlimited", value=0)
    result = solve(players, fixtures, [1], squad, transfers)
    assert result["status"] == "Optimal"
    assert result["transfers"]["total"] > 2
    assert result["transfers"]["paid"] == 0
    assert net_score(result) == pytest.approx(
        reference_optimum(players, fixtures, [1], squad, 300, free_first=99)
    )


def test_free_transfers_apply_to_next_events_gameweek_only():
    players, fixtures = make_league(5)
    squad = legal_squad(players)
    transfers = Transfers(bank=300, cost=1000, limit=2, made=2, status="cost", value=0)
    # The 0 remaining free transfers belong to gameweek 2; planning only gameweek 3 gets a fresh 2
    later = solve(players, fixtures, [3, 4], squad, transfers, next_event=1)
    assert later["transfers"]["paid"] == 0
    assert net_score(later) == pytest.approx(
        reference_optimum(players, fixtures, [3, 4], squad, 300, free_first=2)
    )


def test_stopping_at_gap_limit_is_not_labelled_optimal():
    players, fixtures = make_league(0)
    config = CONFIG._replace(RELATIVE_GAP_LIMIT=1.0)
    optimizer = TeamOptimizer(players, fixtures, PHASES, TEAMS, "form", config=config)
    result = optimizer.optimize([1, 2, 3, 4])
    assert result["optimality_gap"] > 0
    assert result["status"] == "Feasible"
    assert result["solver_stats"]["stop_reason"] == "gap_limit"


def test_rebought_player_costs_current_price():
    # Player 1 sells for 100 but costs 300 to buy back. The only affordable upgrade on day 1
    # is selling him for player 2; buying him back on day 2 must not be valued at 100.
    fillers = [player(10 + t, 1, t, 200, 100) for t in range(1, 5)]
    fillers += [player(20 + t, 2, t, 100, 100) for t in range(1, 6)]
    players = fillers + [player(1, 1, 6, 300, 300), player(2, 1, 7, 400, 100)]
    fixtures = [
        {"event": e, "team_h": h, "team_a": a}
        for e, games in {1: [(1, 2), (3, 4), (5, 7)], 2: [(1, 2), (3, 4), (5, 6)]}.items()
        for h, a in games
    ]
    squad = [{"element": 1, "selling_price": 100}] + [
        {"element": p["id"], "selling_price": 50} for p in fillers
    ]
    transfers = Transfers(bank=0, cost=1000, limit=2, made=0, status="cost", value=0)
    result = solve(players, fixtures, [1, 2], squad, transfers)
    rosters = {e: {p["id"] for p in r} for e, r in result["squad_by_event"].items()}
    assert 2 in rosters["1"] and 1 not in rosters["1"]
    assert 1 not in rosters["2"]
