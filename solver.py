from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from ortools.sat.python import cp_model
import os
import threading
import time
from typing import NamedTuple, Optional


def _log(message: str) -> None:
    print(message, flush=True)


class Position(Enum):
    BACK_COURT = 1
    FRONT_COURT = 2

    @classmethod
    @lru_cache(maxsize=None)  # Cache since we have limited positions
    def from_element_type(cls, element_type: int) -> "Position":
        """Convert API element_type to Position enum."""
        return cls.BACK_COURT if element_type == 1 else cls.FRONT_COURT

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    position: Position
    team: int
    form: float
    points_per_game: float
    now_cost: int
    selling_price: Optional[int] = None
    availability: float = 1.0

    def get_raw_score(self, metric: str) -> float:
        return self.form if metric == "form" else self.points_per_game

    def get_score(self, metric: str) -> float:
        """Expected score for a game, discounted by the chance of playing."""
        return self.get_raw_score(metric) * self.availability

    def get_cost(self) -> int:
        return self.selling_price if self.selling_price is not None else self.now_cost


@dataclass(frozen=True)
class Fixture:
    event: int
    team_h: int
    team_a: int


@dataclass(frozen=True)
class Phase:
    id: int
    start_event: int
    stop_event: int


class OptimizationConfig(NamedTuple):
    """Configuration parameters for optimization"""

    SQUAD_SIZE: int = 10
    FRONT_COURT_COUNT: int = 5
    BACK_COURT_COUNT: int = 5
    STARTING_PLAYERS: int = 5
    MAX_STARTERS_PER_POSITION: int = 3
    MAX_PLAYERS_PER_TEAM: int = 2
    FREE_TRANSFERS: int = 2
    TRANSFER_PENALTY: int = 1000  # API points are x10, so this is the in-game -100
    INITIAL_BUDGET: int = 1000
    SOLVER_TIMEOUT: int = 300  # seconds
    SOLVER_WORKERS: int = 8
    RELATIVE_GAP_LIMIT: float = 0.01  # stop once within 1% of the best possible score
    STALL_SECONDS: Optional[int] = 60  # stop after this long without a better squad
    POINTS_SCALE: int = 1000


@dataclass
class Transfers:
    bank: int
    cost: int
    limit: Optional[int]
    made: int
    status: str
    value: int


class _SearchProgress(cp_model.CpSolverSolutionCallback):
    """Logs each better squad, and a tighter bound every few seconds."""

    def __init__(self, scale: int):
        super().__init__()
        self._scale = scale
        self._reported = 0
        self._saw_bound = False
        self._last_print = 0.0
        self._objective: Optional[float] = None
        self._bound: Optional[float] = None
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._last_improvement: Optional[float] = None

    def seconds_since_improvement(self) -> Optional[float]:
        with self._lock:
            if self._last_improvement is None:
                return None
            return time.monotonic() - self._last_improvement

    def on_solution_callback(self) -> None:
        with self._lock:
            self._last_improvement = time.monotonic()
            objective = self.ObjectiveValue()
            self._bound = self.BestObjectiveBound()
            # The empty lineup scores ~0 and is not a squad worth reporting.
            if abs(objective) < 0.5 * self._scale:
                return
            self._objective = objective
            self._reported += 1
            self._last_print = time.monotonic()
            self._log_state(f"solution {self._reported}")

    def on_bound(self, bound: float) -> None:
        with self._lock:
            # The first callback is the loose domain ceiling, not search progress.
            if not self._saw_bound:
                self._saw_bound = True
                self._bound = bound
                return
            self._bound = bound
            now = time.monotonic()
            if now - self._last_print < 5:
                return
            self._last_print = now
            label = "no squad yet" if self._objective is None else "tighter bound"
            self._log_state(label)

    def _log_state(self, what: str) -> None:
        elapsed = time.monotonic() - self._started
        bound = None if self._bound is None else self._bound / self._scale
        bound_text = "n/a" if bound is None else f"{bound:.1f}"
        if self._objective is None:
            _log(f"Search {elapsed:.1f}s: {what}, bound {bound_text}")
            return
        objective = self._objective / self._scale
        gap = 0.0 if bound is None else max(0.0, bound - objective)
        _log(
            f"Search {elapsed:.1f}s: {what}, "
            f"objective {objective:.1f}, bound {bound_text}, gap {gap:.1f}"
        )


class _StallWatchdog(threading.Thread):
    """Stops the search once no better squad has been found for a while."""

    def __init__(self, solver: cp_model.CpSolver, progress: _SearchProgress, seconds: float):
        super().__init__(daemon=True)
        self._solver = solver
        self._progress = progress
        self._seconds = seconds
        self.done = threading.Event()
        self.fired = False

    def run(self) -> None:
        while not self.done.wait(0.5):
            idle = self._progress.seconds_since_improvement()
            if idle is not None and idle > self._seconds:
                self.fired = True
                self._solver.StopSearch()
                return


def player_availability(player: dict) -> float:
    """Probability that a player plays, based on the API's status fields."""
    chance = player.get("chance_of_playing_next_round")
    if chance is not None:
        return chance / 100
    if player["status"] == "a":
        return 1.0
    if player["status"] == "d":
        return 0.5
    return 0.0


class TeamOptimizer:
    def __init__(
        self,
        players: list[dict],
        fixtures: list[dict],
        phases: list[dict],
        teams: list[dict],
        scoring_metric: str = "form",
        current_squad: Optional[list[dict]] = None,
        config: Optional[OptimizationConfig] = None,
        transfers: Optional[Transfers] = None,
        next_event: Optional[int] = None,
        wildcard: bool = False,
    ):
        print(f"transfers :>> {transfers}")
        self.config = config or OptimizationConfig()
        self.scoring_metric = scoring_metric
        self.current_squad = current_squad or []
        self.current_squad_ids = {p["element"] for p in self.current_squad}
        self.next_event = next_event

        # A null limit means this gameday has no transfer cap. Later gamedays keep the normal allowance.
        self.free_first_gameday = wildcard or (
            transfers is not None and transfers.limit is None
        )
        self.free_transfers = (
            max(0, transfers.limit - transfers.made)
            if transfers is not None and transfers.limit is not None
            else self.config.FREE_TRANSFERS
        )

        self.team_names = {t["id"]: t["name"] for t in teams}
        self.team_short_names = {t["id"]: t["short_name"] for t in teams}

        self.budget = (
            sum(p["selling_price"] for p in self.current_squad)
            + (transfers.bank if transfers else 0)
            if self.current_squad
            else self.config.INITIAL_BUDGET
        )

        self.players = self._process_players(players)
        self.player_map = {p.id: p for p in self.players}

        self.fixtures = self._process_fixtures(fixtures)
        self.phases = self._process_phases(phases)

        self.players_by_team = defaultdict(list)
        self.players_by_position = defaultdict(list)
        for player in self.players:
            self.players_by_team[player.team].append(player)
            self.players_by_position[player.position].append(player)

        self._playing_teams_cache = {}
        self._gameweek_cache = {}

    def _process_players(self, players: list[dict]) -> list[Player]:
        """Process raw player data into Player objects."""
        selling_prices = {p["element"]: p["selling_price"] for p in self.current_squad}

        return [
            Player(
                id=p["id"],
                name=p["web_name"],
                position=Position.from_element_type(p["element_type"]),
                team=p["team"],
                form=float(p["form"] or 0),
                points_per_game=float(p["points_per_game"] or 0),
                now_cost=p["now_cost"],
                selling_price=selling_prices.get(p["id"]),
                availability=player_availability(p),
            )
            for p in players
            if p["status"] in ["a", "d"] or p["id"] in selling_prices
        ]

    def _process_fixtures(self, fixtures: list[dict]) -> list[Fixture]:
        """Process fixtures with field filtering."""
        fixture_fields = {field.name for field in fields(Fixture)}
        return [
            Fixture(**{k: v for k, v in f.items() if k in fixture_fields})
            for f in fixtures
        ]

    def _process_phases(self, phases: list[dict]) -> list[Phase]:
        """Process phases with field filtering, skipping the "Overall" phase."""
        phase_fields = {field.name for field in fields(Phase)}
        return [
            Phase(**{k: v for k, v in p.items() if k in phase_fields})
            for p in phases[1:]
        ]

    def _get_gameweek_for_event(self, event_id: int) -> Optional[int]:
        """Get the gameweek (phase) ID for a given event ID."""
        if event_id in self._gameweek_cache:
            return self._gameweek_cache[event_id]

        gameweek = next(
            (p.id for p in self.phases if p.start_event <= event_id <= p.stop_event),
            None,
        )
        self._gameweek_cache[event_id] = gameweek
        return gameweek

    def _get_playing_teams(self, event_id: int) -> frozenset:
        """Get set of teams playing in a given event."""
        if event_id not in self._playing_teams_cache:
            self._playing_teams_cache[event_id] = frozenset(
                team_id
                for fixture in self.fixtures
                if fixture.event == event_id
                for team_id in (fixture.team_h, fixture.team_a)
            )
        return self._playing_teams_cache[event_id]

    def _scale_points(self, points: float) -> int:
        """Scale floating point scores to integers for CP-SAT solver."""
        return round(points * self.config.POINTS_SCALE)

    def _process_player_for_output(self, player: Player) -> dict:
        """Helper to convert player data to user-friendly format"""
        return {
            "id": player.id,
            "name": player.name,
            "team": self.team_names.get(player.team),
            "team_short": self.team_short_names.get(player.team),
            "position": player.position,
            "points": player.get_raw_score(self.scoring_metric),
            "availability": player.availability,
            "cost": player.get_cost(),
        }

    def _current_gameweek(self, sorted_events: list[int]) -> Optional[int]:
        """Gameweek whose remaining free transfers are given by the transfers info."""
        return self._get_gameweek_for_event(
            self.next_event if self.next_event is not None else sorted_events[0]
        )

    def _free_transfers_for(self, gameweek: Optional[int], current_gameweek) -> int:
        if gameweek == current_gameweek:
            return self.free_transfers
        return self.config.FREE_TRANSFERS

    def _is_free_transfer_event(self, index: int) -> bool:
        """Transfers on the first gameday are free when building a new squad, on a wildcard, or when that gameday has no limit."""
        return index == 0 and (not self.current_squad_ids or self.free_first_gameday)

    def _group_chargeable_events(self, sorted_events: list[int]) -> dict:
        """Group events whose transfers count against the weekly free allowance."""
        gameweeks = defaultdict(list)
        for i, event_id in enumerate(sorted_events):
            if not self._is_free_transfer_event(i):
                gameweeks[self._get_gameweek_for_event(event_id)].append(event_id)
        return gameweeks

    def _build_model(self, sorted_events: list[int]) -> cp_model.CpModel:
        model = cp_model.CpModel()
        cfg = self.config
        self.in_squad = {}
        self.transfer_in = {}
        self.transfer_out = {}
        self.starts = {}
        objective_terms = []

        # ===== 1. ROSTER EVOLUTION =====
        _log(
            f"Model 1/5: roster links "
            f"({len(self.players)} players, {len(sorted_events)} gamedays)"
        )
        for player in self.players:
            for i, event_id in enumerate(sorted_events):
                key = (player.id, event_id)
                x = self.in_squad[key] = model.NewBoolVar(f"in_squad_{player.id}_{event_id}")
                t_in = self.transfer_in[key] = model.NewBoolVar(f"in_{player.id}_{event_id}")
                t_out = self.transfer_out[key] = model.NewBoolVar(f"out_{player.id}_{event_id}")
                model.Add(t_in + t_out <= 1)

                if i > 0:
                    prev = self.in_squad[(player.id, sorted_events[i - 1])]
                    model.Add(x == prev + t_in - t_out)
                elif self.current_squad_ids:
                    initial = 1 if player.id in self.current_squad_ids else 0
                    model.Add(x == initial + t_in - t_out)
                else:
                    model.Add(t_in == 0)
                    model.Add(t_out == 0)

        # ===== 2. SQUAD RULES FOR EVERY GAMEDAY =====
        _log("Model 2/5: squad size, position, and team limits")
        position_counts = {
            Position.BACK_COURT: cfg.BACK_COURT_COUNT,
            Position.FRONT_COURT: cfg.FRONT_COURT_COUNT,
        }
        for event_id in sorted_events:
            for position, count in position_counts.items():
                model.Add(
                    sum(self.in_squad[(p.id, event_id)] for p in self.players_by_position[position])
                    == count
                )
            for team_players in self.players_by_team.values():
                model.Add(
                    sum(self.in_squad[(p.id, event_id)] for p in team_players)
                    <= cfg.MAX_PLAYERS_PER_TEAM
                )

        # ===== 3. BUDGET =====
        # A current-squad player is valued at his selling price only while held
        # continuously; if sold and bought back he costs now_cost again.
        priced = sum(1 for p in self.players if p.selling_price is not None)
        _log(f"Model 3/5: budget ({priced} players keep a selling price)")
        kept = {}
        for player in self.players:
            if player.selling_price is None:
                continue
            for i, event_id in enumerate(sorted_events):
                x = self.in_squad[(player.id, event_id)]
                k = kept[(player.id, event_id)] = model.NewBoolVar(f"kept_{player.id}_{event_id}")
                if i == 0:
                    model.Add(k == x)
                else:
                    k_prev = kept[(player.id, sorted_events[i - 1])]
                    model.Add(k <= x)
                    model.Add(k <= k_prev)
                    model.Add(k >= x + k_prev - 1)

        for event_id in sorted_events:
            model.Add(
                sum(self.in_squad[(p.id, event_id)] * p.now_cost for p in self.players)
                - sum(
                    kept[(p.id, event_id)] * (p.now_cost - p.selling_price)
                    for p in self.players
                    if p.selling_price is not None
                )
                <= self.budget
            )

        # ===== 4. STARTING LINE-UP (only starters score) =====
        _log("Model 4/5: starting line-ups and scores")
        for event_id in sorted_events:
            playing_teams = self._get_playing_teams(event_id)
            day_starts = defaultdict(list)
            for player in self.players:
                if player.team not in playing_teams:
                    continue
                s = self.starts[(player.id, event_id)] = model.NewBoolVar(
                    f"start_{player.id}_{event_id}"
                )
                model.Add(s <= self.in_squad[(player.id, event_id)])
                day_starts[player.position].append(s)
                objective_terms.append(
                    s * self._scale_points(player.get_score(self.scoring_metric))
                )
            model.Add(sum(s for group in day_starts.values() for s in group) <= cfg.STARTING_PLAYERS)
            for group in day_starts.values():
                model.Add(sum(group) <= cfg.MAX_STARTERS_PER_POSITION)

        # ===== 5. TRANSFER PENALTIES (per gameweek) =====
        current_gameweek = self._current_gameweek(sorted_events)
        chargeable = self._group_chargeable_events(sorted_events)
        weeks = len(chargeable)
        week_word = "gameweek" if weeks == 1 else "gameweeks"
        _log(f"Model 5/5: transfer penalties ({weeks} {week_word})")
        for gameweek, gw_events in chargeable.items():
            total = sum(
                self.transfer_in[(p.id, e)] for p in self.players for e in gw_events
            )
            excess = model.NewIntVar(
                0, cfg.SQUAD_SIZE * len(gw_events), f"excess_transfers_{gameweek}"
            )
            model.Add(excess >= total - self._free_transfers_for(gameweek, current_gameweek))
            objective_terms.append(-excess * self._scale_points(cfg.TRANSFER_PENALTY))

        model.Maximize(sum(objective_terms))
        _log(
            f"Model ready: {len(self.in_squad)} squad slots, "
            f"{len(self.starts)} possible starters, "
            f"{len(objective_terms)} objective terms"
        )
        return model

    def _add_solver_hints(self, model: cp_model.CpModel, sorted_events: list[int]):
        """Hint the no-transfer plan, which is always feasible for an existing squad."""
        if not self.current_squad_ids:
            _log("Hints: skipped (no current squad)")
            return
        _log(f"Hints: keep the current {len(self.current_squad_ids)} and make no transfers")
        for player in self.players:
            held = 1 if player.id in self.current_squad_ids else 0
            for event_id in sorted_events:
                model.AddHint(self.in_squad[(player.id, event_id)], held)
                model.AddHint(self.transfer_in[(player.id, event_id)], 0)
                model.AddHint(self.transfer_out[(player.id, event_id)], 0)

    def optimize(self, event_ids: list[int]) -> dict:
        """Optimize team selection for given events."""
        if not event_ids:
            return {"error": "No events provided for optimization"}

        missing = self.current_squad_ids - self.player_map.keys()
        if missing:
            return {
                "status": "Error",
                "solver": "cpsat",
                "error": f"Current squad players missing from player pool: {sorted(missing)}",
            }

        sorted_events = sorted(set(event_ids))
        first, last = sorted_events[0], sorted_events[-1]
        span = f"{first}" if first == last else f"{first}-{last}"
        if not self.current_squad_ids:
            transfer_note = "new squad, so transfers on the first gameday are free"
        elif self.free_first_gameday:
            transfer_note = "no transfer limit on the first gameday"
        else:
            transfer_note = f"{self.free_transfers} free transfers left this gameweek"
        _log(
            f"Optimizing gamedays {span} ({len(sorted_events)} total) "
            f"for {len(self.players)} players, budget {self.budget}, {transfer_note}"
        )

        try:
            _log("Creating optimization model...")
            build_started = time.perf_counter()
            model = self._build_model(sorted_events)
            self._add_solver_hints(model, sorted_events)
            build_time = time.perf_counter() - build_started

            workers = min(self.config.SOLVER_WORKERS, os.cpu_count() or 1)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.config.SOLVER_TIMEOUT
            solver.parameters.num_workers = workers
            solver.parameters.relative_gap_limit = self.config.RELATIVE_GAP_LIMIT
            progress = _SearchProgress(self.config.POINTS_SCALE)
            solver.best_bound_callback = progress.on_bound

            stall = self.config.STALL_SECONDS
            _log(
                f"Solving (timeout {self.config.SOLVER_TIMEOUT}s, {workers} workers, "
                f"stop within {self.config.RELATIVE_GAP_LIMIT:.1%} of best possible"
                + (f" or after {stall}s without a better squad). " if stall else "). ")
                + "A line prints for each better squad, and every few seconds as the bound tightens."
            )
            watchdog = _StallWatchdog(solver, progress, stall) if stall else None
            if watchdog:
                watchdog.start()
            try:
                status = solver.Solve(model, progress)
            finally:
                if watchdog:
                    watchdog.done.set()
                    watchdog.join()
            status_name = solver.status_name(status)
            wall = solver.WallTime()
            found = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            # CP-SAT also reports OPTIMAL when it stops at the gap limit.
            proven = found and solver.BestObjectiveBound() - solver.ObjectiveValue() < 1
            if proven:
                stop_reason = "proven_optimal"
            elif watchdog and watchdog.fired:
                stop_reason = "no_improvement"
            elif status == cp_model.OPTIMAL:
                stop_reason = "gap_limit"
            else:
                stop_reason = "timeout"
            stats = {
                "stop_reason": stop_reason,
                "players": len(self.players),
                "gamedays": len(sorted_events),
                "workers": workers,
                "booleans": solver.NumBooleans(),
                "branches": solver.NumBranches(),
                "conflicts": solver.NumConflicts(),
                "build_seconds": round(build_time, 2),
                "solve_seconds": round(wall, 2),
            }
            _log(
                f"Solver stats: stopped by {stop_reason}, {stats['booleans']} booleans, "
                f"{stats['branches']} branches, {stats['conflicts']} conflicts, "
                f"build {build_time:.1f}s, solve {wall:.1f}s"
            )
            if found:
                scale = self.config.POINTS_SCALE
                objective = solver.ObjectiveValue() / scale
                bound = solver.BestObjectiveBound() / scale
                _log(
                    f"Solver status: {status_name} in {wall:.1f}s "
                    f"(objective {objective:.1f}, bound {bound:.1f}, "
                    f"gap {max(0.0, bound - objective):.1f})"
                )
                result = self._process_solution(solver, proven, sorted_events)
                result["solver_stats"] = stats
                return result
            _log(f"Solver status: {status_name} in {wall:.1f}s")
            return {
                "status": "Infeasible",
                "solver": "cpsat",
                "error": f"No solution found by the solver ({status_name}).",
                "solver_stats": stats,
            }

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {
                "status": "Error",
                "solver": "cpsat",
                "error": f"An exception occurred: {str(e)}",
            }

    def _calculate_true_gameday_score(
        self, roster_player_ids, event_id: int
    ) -> tuple[float, list[dict]]:
        """
        Calculates the maximized score for a given 10-player roster
        by finding the optimal 3-2 or 2-3 starting lineup for the gameday.
        """
        playing_teams = self._get_playing_teams(event_id)
        by_position = {Position.BACK_COURT: [], Position.FRONT_COURT: []}

        for p_id in roster_player_ids:
            player = self.player_map.get(p_id)
            if player and player.team in playing_teams:
                by_position[player.position].append(
                    {
                        "id": player.id,
                        "name": player.name,
                        "team": self.team_short_names.get(player.team),
                        "position": player.position,
                        "points": player.get_score(self.scoring_metric),
                    }
                )

        back = sorted(by_position[Position.BACK_COURT], key=lambda x: x["points"], reverse=True)
        front = sorted(by_position[Position.FRONT_COURT], key=lambda x: x["points"], reverse=True)

        best_score, best_starters = 0.0, []
        for n_back in (3, 2):
            starters = back[:n_back] + front[: self.config.STARTING_PLAYERS - n_back]
            score = sum(p["points"] for p in starters)
            if score > best_score or not best_starters:
                best_score, best_starters = score, starters
        return best_score, best_starters

    def _process_solution(
        self, solver: cp_model.CpSolver, proven: bool, sorted_events: list[int]
    ) -> dict:
        """Process optimization solution into result format."""
        _log(f"Reading the solution across {len(sorted_events)} gamedays...")
        value = lambda var: solver.Value(var) == 1

        rosters = {
            e: [p.id for p in self.players if value(self.in_squad[(p.id, e)])]
            for e in sorted_events
        }

        daily_starters = {}
        starter_ids = {}
        total_true_score = 0.0
        for event_id in sorted_events:
            score, starters = self._calculate_true_gameday_score(rosters[event_id], event_id)
            total_true_score += score
            daily_starters[event_id] = {"score": score, "starters": starters}
            starter_ids[event_id] = {s["id"] for s in starters}

        # ----- Transfers -----
        transfers_by_event = {e: [] for e in sorted_events}
        current_gameweek = self._current_gameweek(sorted_events)
        gameweek_counts = defaultdict(int)
        paid_transfers_total = 0

        for i, event_id in enumerate(sorted_events):
            free_event = self._is_free_transfer_event(i)
            gameweek = self._get_gameweek_for_event(event_id)
            free_limit = self._free_transfers_for(gameweek, current_gameweek)

            for position in (Position.BACK_COURT, Position.FRONT_COURT):
                ins = [
                    p for p in self.players_by_position[position]
                    if value(self.transfer_in[(p.id, event_id)])
                ]
                outs = [
                    p for p in self.players_by_position[position]
                    if value(self.transfer_out[(p.id, event_id)])
                ]
                for in_player, out_player in zip(ins, outs):
                    if free_event:
                        transfer_number = len(transfers_by_event[event_id]) + 1
                        is_paid = False
                    else:
                        gameweek_counts[gameweek] += 1
                        transfer_number = gameweek_counts[gameweek]
                        is_paid = transfer_number > free_limit
                    paid_transfers_total += is_paid
                    penalty = self.config.TRANSFER_PENALTY if is_paid else 0

                    # Compare over the days the incoming player stays in the squad
                    window = []
                    for e in sorted_events[i:]:
                        if in_player.id not in rosters[e]:
                            break
                        window.append(e)
                    gained = [
                        in_player.get_score(self.scoring_metric)
                        for e in window
                        if in_player.id in starter_ids[e]
                    ]
                    lost = [
                        out_player.get_score(self.scoring_metric)
                        for e in window
                        if out_player.team in self._get_playing_teams(e)
                    ]

                    transfers_by_event[event_id].append(
                        {
                            "phase": gameweek,
                            "transfer_number": transfer_number,
                            "is_paid": is_paid,
                            "penalty": penalty,
                            "points_gained": sum(gained),
                            "points_lost": sum(lost),
                            "net_points": sum(gained) - sum(lost) - penalty,
                            "games_gained": len(gained),
                            "games_lost": len(lost),
                            "in": self._process_player_for_output(in_player),
                            "out": self._process_player_for_output(out_player),
                        }
                    )

        transfers_by_event = {
            str(event_id): data for event_id, data in transfers_by_event.items() if data
        }
        final_squad_players = [
            self._process_player_for_output(self.player_map[p_id])
            for p_id in rosters[sorted_events[-1]]
        ]
        scale = self.config.POINTS_SCALE
        objective = solver.ObjectiveValue() / scale
        bound = solver.BestObjectiveBound() / scale
        label = "Optimal" if proven else "Feasible"
        n_transfers = sum(len(t) for t in transfers_by_event.values())
        _log(
            f"Result: {label}, score {total_true_score:.1f} "
            f"over {len(sorted_events)} gamedays, "
            f"{n_transfers} transfers ({paid_transfers_total} paid)"
        )

        return {
            "status": label,
            "squad": final_squad_players,
            "squad_by_event": {
                str(e): [self._process_player_for_output(self.player_map[p_id]) for p_id in ids]
                for e, ids in rosters.items()
            },
            "total_cost": sum(p["cost"] for p in final_squad_players),
            "solver_objective_value": objective,
            "optimality_gap": max(0.0, bound - objective),
            "true_gameweek_score": total_true_score,
            "total_games": sum(len(d["starters"]) for d in daily_starters.values()),
            "daily_starters": daily_starters,
            "average_points_per_day": total_true_score / len(sorted_events),
            "transfers": {
                "by_event": transfers_by_event,
                "total": n_transfers,
                "paid": paid_transfers_total,
                "cost": paid_transfers_total * self.config.TRANSFER_PENALTY,
                "total_net_points": sum(
                    t["net_points"] for ts in transfers_by_event.values() for t in ts
                ),
            },
        }
