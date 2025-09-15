from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from ortools.sat.python import cp_model
from typing import Dict, NamedTuple, Optional, List, Tuple
import copy
import os


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

    def get_score(self, metric: str) -> float:
        return self.form if metric == "form" else self.points_per_game

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
    MAX_PLAYERS_PER_TEAM: int = 2
    FREE_TRANSFERS: int = 2
    TRANSFER_PENALTY: int = 1000
    SOLVER_TIMEOUT: int = 300  # seconds


@dataclass
class Transfers:
    bank: int
    cost: int
    limit: int
    made: int
    status: str
    value: int


class TeamOptimizer:
    def __init__(
        self,
        players: List[Dict],
        fixtures: List[Dict],
        phases: List[Dict],
        teams: List[Dict],
        scoring_metric: str = "form",
        current_squad: Optional[List[Dict]] = None,
        config: Optional[OptimizationConfig] = None,
        transfers: Optional[Transfers] = None,
    ):
        self.config = config or OptimizationConfig()
        self.scoring_metric = scoring_metric

        # Calculate free transfers from transfers info if available
        self.free_transfers = (
            transfers.limit - transfers.made
            if transfers is not None
            else self.config.FREE_TRANSFERS
        )
        # self.free_transfers = 2

        # Pre-compute team lookups before processing players
        self.team_lookup = {t["id"]: t for t in teams}
        self.team_names = {t["id"]: t["name"] for t in teams}
        self.team_short_names = {t["id"]: t["short_name"] for t in teams}

        # Calculate budget once
        self.budget = (
            sum(p["selling_price"] for p in current_squad) + transfers.bank
            if current_squad and transfers
            else 1000
        )

        # Process data in optimal order
        self.players = self._process_players(players, current_squad)
        self.fixtures = self._process_fixtures(fixtures)
        self.phases = self._process_phases(phases)

        # Pre-compute player lookups for optimization
        self.players_by_team = defaultdict(list)
        self.players_by_position = defaultdict(list)
        for player in self.players:
            self.players_by_team[player.team].append(player)
            self.players_by_position[player.position].append(player)

        # Cache for expensive computations
        self._playing_teams_cache = {}
        self._gameweek_cache = {}

    def _process_players(
        self, players: List[Dict], current_squad: Optional[List[Dict]]
    ) -> List[Player]:
        """Process raw player data into Player objects."""
        current_squad_map = {
            p["element"]: p["selling_price"] for p in (current_squad or [])
        }

        return [
            Player(
                id=p["id"],
                name=p["web_name"],
                position=Position.from_element_type(p["element_type"]),
                team=p["team"],
                form=float(p["form"] or 0),
                points_per_game=float(p["points_per_game"] or 0),
                now_cost=p["now_cost"],
                selling_price=current_squad_map.get(p["id"]),
            )
            for p in players
            if p["status"] in ["a", "d"] or p["id"] in current_squad_map
        ]

    def _process_fixtures(self, fixtures: List[Dict]) -> List[Fixture]:
        """Process fixtures with field filtering."""
        fixture_fields = {field.name for field in fields(Fixture)}
        return [
            Fixture(**{k: v for k, v in f.items() if k in fixture_fields})
            for f in fixtures
        ]

    def _process_phases(self, phases: List[Dict]) -> List[Phase]:
        """Process phases with field filtering."""
        phase_fields = {field.name for field in fields(Phase)}
        return [
            Phase(**{k: v for k, v in p.items() if k in phase_fields})
            for p in phases[1:]
        ]

    def _get_gameweek_for_event(self, event_id: int) -> Optional[int]:
        """Get the gameweek (phase) ID for a given event ID."""
        if event_id in self._gameweek_cache:
            return self._gameweek_cache[event_id]

        for phase in self.phases:
            if phase.start_event <= event_id <= phase.stop_event:
                self._gameweek_cache[event_id] = phase.id
                return phase.id

        self._gameweek_cache[event_id] = None
        return None

    def _get_playing_teams(self, event_id: int) -> frozenset:
        """Get set of teams playing in a given event."""
        if event_id in self._playing_teams_cache:
            return self._playing_teams_cache[event_id]

        teams = frozenset(
            team_id
            for fixture in self.fixtures
            if fixture.event == event_id
            for team_id in (fixture.team_h, fixture.team_a)
        )

        self._playing_teams_cache[event_id] = teams
        return teams

    def _scale_points(self, points: float) -> int:
        """Scale floating point scores to integers for CP-SAT solver."""
        return int(points * 1000)

    def _process_player_for_output(self, player):
        """Helper to convert player data to user-friendly format"""
        return {
            "id": player.id,
            "name": player.name,
            "team": self.team_names.get(player.team),
            "team_short": self.team_short_names.get(player.team),
            "position": player.position,
            "points": (
                player.form if self.scoring_metric == "form" else player.points_per_game
            ),
            "cost": player.selling_price if player.selling_price else player.now_cost,
        }

    def _create_optimization_model(
        self, event_ids: List[int], current_squad: Optional[List[Dict]]
    ) -> Tuple[cp_model.CpModel, Dict, Dict, Dict]:
        """Create CP-SAT model and variables."""
        model = cp_model.CpModel()

        # Pre-allocate dictionaries with expected size
        player_count = len(self.players)
        event_count = len(event_ids)
        gameweeks = {self._get_gameweek_for_event(e) for e in event_ids}
        gameweek_count = len(gameweeks)

        player_vars = {}  # Boolean vars for squad selection
        starter_vars = {}  # Boolean vars for starting lineup each event
        transfer_vars = {}  # Boolean vars for transfers

        # Batch variable creation for better performance
        for player in self.players:
            # Squad selection vars
            player_vars[player.id] = model.NewBoolVar(f"player_{player.id}")

            # Starting lineup vars for each event
            for event_id in event_ids:
                starter_vars[(player.id, event_id)] = model.NewBoolVar(
                    f"starter_{player.id}_{event_id}"
                )

            # Transfer vars for each gameweek
            for gw in gameweeks:
                if gw is not None:  # Skip None values
                    transfer_vars[(player.id, gw)] = model.NewBoolVar(
                        f"transfer_{player.id}_{gw}"
                    )

        # Add symmetry breaking constraints for similar players
        self._add_symmetry_breaking(model, player_vars)

        return model, player_vars, starter_vars, transfer_vars

    def _add_symmetry_breaking(self, model, player_vars):
        """Add symmetry breaking constraints as a separate method for clarity"""
        for position in [Position.FRONT_COURT, Position.BACK_COURT]:
            position_players = sorted(
                self.players_by_position[position], key=lambda p: p.id
            )

            # Group players by similar stats and costs
            similar_players = defaultdict(list)
            for player in position_players:
                key = (
                    round(player.get_score(self.scoring_metric), 1),
                    player.get_cost(),
                )
                similar_players[key].append(player)

            # Add constraints only for groups with multiple similar players
            for players in similar_players.values():
                if len(players) > 1:
                    for i in range(len(players) - 1):
                        model.Add(
                            player_vars[players[i].id] >= player_vars[players[i + 1].id]
                        )

    def _add_constraints(
        self,
        model: cp_model.CpModel,
        player_vars: Dict,
        starter_vars: Dict,
        transfer_vars: Dict,
        event_ids: List[int],
        current_squad: Optional[List[Dict]],
        wildcard: Optional[bool] = False,
    ) -> None:
        """Add constraints to the optimization model."""
        objective_terms = []
        sorted_events = sorted(event_ids)

        # ===== 1. SQUAD COMPOSITION CONSTRAINTS =====
        # Basic squad size constraint
        model.Add(sum(player_vars.values()) == self.config.SQUAD_SIZE)

        # Position balance constraints
        for position, players in self.players_by_position.items():
            count = (
                self.config.FRONT_COURT_COUNT
                if position == Position.FRONT_COURT
                else self.config.BACK_COURT_COUNT
            )
            model.Add(sum(player_vars[p.id] for p in players) == count)

        # Budget constraint for final squad
        model.Add(
            sum(player_vars[p.id] * p.get_cost() for p in self.players) <= self.budget
        )

        # Team limit constraint
        for team_players in self.players_by_team.values():
            model.Add(
                sum(player_vars[p.id] for p in team_players)
                <= self.config.MAX_PLAYERS_PER_TEAM
            )

        # ===== 2. TRANSFER CONSTRAINTS (if applicable) =====
        if current_squad and not wildcard:
            objective_terms.extend(
                self._add_transfer_constraints(
                    model, player_vars, event_ids, sorted_events, current_squad
                )
            )

        # ===== 3. STARTING LINEUP CONSTRAINTS =====
        for event_id in event_ids:
            objective_terms.extend(
                self._add_lineup_constraints(
                    model, player_vars, starter_vars, event_id, current_squad, wildcard
                )
            )

        return objective_terms

    def _add_transfer_constraints(
        self, model, player_vars, event_ids, sorted_events, current_squad
    ):
        """Add transfer-related constraints to the model."""
        objective_terms = []
        current_players = {p["element"] for p in current_squad}

        # Group events by phase - do this once
        phases = defaultdict(list)
        for event_id in event_ids:
            phase = self._get_gameweek_for_event(event_id)
            phases[phase].append(event_id)

        # Pre-allocate all variables at once
        self.event_transfer_in_vars = {}
        self.event_transfer_out_vars = {}
        self.player_in_squad_vars = {}

        # Batch variable creation
        for player in self.players:
            for event_id in sorted_events:
                # Squad status variable
                self.player_in_squad_vars[(player.id, event_id)] = model.NewBoolVar(
                    f"player_in_squad_{player.id}_{event_id}"
                )

                # Transfer variables
                self.event_transfer_in_vars[(player.id, event_id)] = model.NewBoolVar(
                    f"transfer_in_{player.id}_{event_id}"
                )
                self.event_transfer_out_vars[(player.id, event_id)] = model.NewBoolVar(
                    f"transfer_out_{player.id}_{event_id}"
                )

        # Set flag to indicate we created event transfer variables
        self._event_transfer_vars_created = True

        # Add constraints in batches for better performance
        self._add_initial_squad_constraints(model, sorted_events, current_players)
        self._add_squad_evolution_constraints(model, sorted_events)
        self._add_final_squad_link_constraints(model, player_vars, sorted_events)
        self._add_transfer_balance_constraints(model, sorted_events)
        self._add_position_balance_constraints(model, sorted_events)
        self._add_budget_constraints(model, sorted_events)

        # Add transfer penalties
        objective_terms.extend(self._add_transfer_penalties(model, phases))

        return objective_terms

    def _add_initial_squad_constraints(self, model, sorted_events, current_players):
        """Add constraints for initial squad status"""
        if not sorted_events:
            return

        first_event = sorted_events[0]

        # Batch constraints by player status
        initial_squad_players = []
        non_squad_players = []

        for player in self.players:
            if player.id in current_players:
                initial_squad_players.append(player)
            else:
                non_squad_players.append(player)

        # Add constraints in batches
        for player in initial_squad_players:
            model.Add(
                self.player_in_squad_vars[(player.id, first_event)]
                == 1 - self.event_transfer_out_vars[(player.id, first_event)]
            )
            model.Add(self.event_transfer_in_vars[(player.id, first_event)] == 0)

        for player in non_squad_players:
            model.Add(
                self.player_in_squad_vars[(player.id, first_event)]
                == self.event_transfer_in_vars[(player.id, first_event)]
            )
            model.Add(self.event_transfer_out_vars[(player.id, first_event)] == 0)

    def _add_lineup_constraints(
        self, model, player_vars, starter_vars, event_id, current_squad, wildcard
    ):
        """Add constraints related to starting lineups."""
        objective_terms = []
        playing_teams = self._get_playing_teams(event_id)

        # Starting lineup size constraint
        model.Add(
            sum(starter_vars[(p.id, event_id)] for p in self.players)
            <= self.config.STARTING_PLAYERS
        )

        # Position balance for starters
        front_court_starters = sum(
            starter_vars[(p.id, event_id)]
            for p in self.players_by_position[Position.FRONT_COURT]
        )
        back_court_starters = sum(
            starter_vars[(p.id, event_id)]
            for p in self.players_by_position[Position.BACK_COURT]
        )

        model.Add(
            front_court_starters + back_court_starters <= self.config.STARTING_PLAYERS
        )
        model.Add(front_court_starters <= 3)
        model.Add(back_court_starters <= 3)

        # Can only start players in the squad for this event
        for player in self.players:
            if current_squad and not wildcard and hasattr(self, "player_in_squad_vars"):
                # Link starters to event-specific squad status
                model.Add(
                    starter_vars[(player.id, event_id)]
                    <= self.player_in_squad_vars[(player.id, event_id)]
                )
            else:
                # If no current squad or using wildcard, link to final squad
                model.Add(starter_vars[(player.id, event_id)] <= player_vars[player.id])

            # Add points to objective if team is playing
            if player.team in playing_teams:
                points = player.get_score(self.scoring_metric)
                objective_terms.append(
                    starter_vars[(player.id, event_id)] * self._scale_points(points)
                )

        return objective_terms

    def optimize(
        self,
        event_ids: List[int],
        current_squad: Optional[List[Dict]] = None,
        wildcard: Optional[bool] = False,
    ) -> Dict:
        """
        Optimize team selection for given events with performance optimizations.
        """
        # Early validation
        if not event_ids:
            return {"error": "No events provided for optimization"}

        # If too many events, use incremental approach
        if len(event_ids) > 7 and current_squad and not wildcard:
            return self._incremental_optimize(event_ids, current_squad)

        try:
            # Create model and variables
            print("Creating optimization model...")
            model, player_vars, starter_vars, transfer_vars = (
                self._create_optimization_model(event_ids, current_squad)
            )

            # Add constraints and get objective terms
            print("Adding constraints and objective terms to the model...")
            objective_terms = self._add_constraints(
                model,
                player_vars,
                starter_vars,
                transfer_vars,
                event_ids,
                current_squad,
                wildcard,
            )

            # Set objective
            model.Maximize(sum(objective_terms))

            print("Configuring the model and adding hints...")
            # Configure solver with optimized parameters
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.config.SOLVER_TIMEOUT
            solver.parameters.linearization_level = 2
            solver.parameters.cp_model_presolve = True
            solver.parameters.cp_model_probing_level = 2

            # For large problems, use more aggressive settings
            if len(event_ids) > 3 or len(self.players) > 200:
                solver.parameters.num_search_workers = min(8, os.cpu_count() or 4)
                solver.parameters.log_search_progress = True

            # Add hints for high-value players
            self._add_solver_hints(model, player_vars, starter_vars, event_ids)

            # Solve the model
            print("Solving...")
            status = solver.Solve(model)

            # print(
            #     "Statuses :>>",
            #     cp_model.UNKNOWN,
            #     cp_model.MODEL_INVALID,
            #     cp_model.FEASIBLE,
            #     cp_model.INFEASIBLE,
            #     cp_model.OPTIMAL,
            # )
            print(f"Solver status :>> {status}")

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._process_solution(
                    solver,
                    player_vars,
                    starter_vars,
                    transfer_vars,
                    event_ids,
                    current_squad,
                )

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            import traceback

            traceback.print_exc()

        return self._fallback_solution(event_ids, current_squad)

    def _add_solver_hints(self, model, player_vars, starter_vars, event_ids):
        """Add hints to guide the solver toward good solutions."""
        # Track which players we've already added hints for
        hinted_players = set()

        # Hint: Start with high-value players
        for player in sorted(
            self.players,
            key=lambda p: p.get_score(self.scoring_metric) / p.get_cost(),
            reverse=True,
        )[
            :20
        ]:  # Focus on top 20 value players
            if player.id not in hinted_players:
                model.AddHint(player_vars[player.id], 1)
                hinted_players.add(player.id)

        # Hint: Start with players from teams that play in most events
        team_event_counts = defaultdict(int)
        for event_id in event_ids:
            playing_teams = self._get_playing_teams(event_id)
            for team_id in playing_teams:
                team_event_counts[team_id] += 1

        # Prioritize players from teams with most games
        for team_id, count in sorted(
            team_event_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count >= len(event_ids) * 0.7:  # Teams playing in at least 70% of events
                for player in self.players_by_team[team_id][
                    :5
                ]:  # Top 5 players per team
                    if player.id not in hinted_players:
                        model.AddHint(player_vars[player.id], 1)
                        hinted_players.add(player.id)

    def _incremental_optimize(self, event_ids, current_squad):
        """Solve the problem incrementally for large event sets."""
        # Sort events chronologically
        sorted_events = sorted(event_ids)

        # Group events by phase - do this once
        phases = defaultdict(list)
        for event_id in sorted_events:
            phase = self._get_gameweek_for_event(event_id)
            phases[phase].append(event_id)

        # Solve one phase at a time
        current_result = None
        all_transfers = {}
        all_starters = {}

        # Sequential optimization for phases
        for phase, phase_events in sorted(phases.items()):
            # Optimize for this phase only
            phase_result = self._optimize_phase(phase_events, current_squad)

            # Update current squad for next phase
            current_squad = [
                {"element": p["id"], "selling_price": p["cost"]}
                for p in phase_result["squad"]
            ]

            # Collect results
            all_transfers.update(phase_result["transfers_by_event"])
            all_starters.update(phase_result["daily_starters"])

            # Save last result
            current_result = phase_result

        # Combine results
        if current_result:
            current_result["transfers_by_event"] = all_transfers
            current_result["daily_starters"] = all_starters
            return current_result

        return self._fallback_solution(event_ids, current_squad)

    def _optimize_phase(self, event_ids, current_squad):
        """Optimize for a single phase with a shorter timeout."""
        # Create a copy of the optimizer with a shorter timeout
        phase_config = copy.deepcopy(self.config)
        phase_config.SOLVER_TIMEOUT = min(
            60, self.config.SOLVER_TIMEOUT
        )  # 60 seconds max per phase

        # Use the standard optimization method but with shorter timeout
        try:
            model, player_vars, starter_vars, transfer_vars = (
                self._create_optimization_model(event_ids, current_squad)
            )

            objective_terms = self._add_constraints(
                model,
                player_vars,
                starter_vars,
                transfer_vars,
                event_ids,
                current_squad,
                False,  # No wildcard for incremental
            )

            model.Maximize(sum(objective_terms))

            # Solve with optimized parameters
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = phase_config.SOLVER_TIMEOUT
            solver.parameters.linearization_level = 2
            solver.parameters.cp_model_presolve = True

            # Add hints
            self._add_solver_hints(model, player_vars, starter_vars, event_ids)

            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._process_solution(
                    solver,
                    player_vars,
                    starter_vars,
                    transfer_vars,
                    event_ids,
                    current_squad,
                )

        except Exception as e:
            print(f"Phase optimization failed: {str(e)}")

        return self._fallback_solution(event_ids, current_squad)

    def _process_solution(
        self,
        solver: cp_model.CpSolver,
        player_vars: Dict,
        starter_vars: Dict,
        transfer_vars: Dict,
        event_ids: List[int],
        current_squad: Optional[List[Dict]],
    ) -> Dict:
        """Process optimization solution into result format."""
        # Pre-compute player lookup for faster access
        player_lookup = {p.id: p for p in self.players}

        # Cache solver.Value calls for better performance
        value_cache = {}

        def get_value(var):
            if var not in value_cache:
                value_cache[var] = solver.Value(var)
            return value_cache[var]

        # Get selected players in one pass
        selected_player_ids = [
            p_id for p_id, var in player_vars.items() if get_value(var) == 1
        ]

        selected_players = [
            self._process_player_for_output(player_lookup[p_id])
            for p_id in selected_player_ids
        ]

        # Process results for all events at once
        starters_by_event = {}
        transfers_by_event = {}

        # Pre-compute event squads if needed
        event_squads = {}
        if current_squad and hasattr(self, "_event_transfer_vars_created"):
            for event_id in event_ids:
                event_squad_ids = [
                    p.id
                    for p in self.players
                    if solver.Value(self.player_in_squad_vars[(p.id, event_id)]) == 1
                ]
                event_squads[event_id] = [
                    self._process_player_for_output(player_lookup[p_id])
                    for p_id in event_squad_ids
                ]

        # Process starters for all events
        for event_id in event_ids:
            # Get squad for this event
            event_squad = event_squads.get(event_id, selected_players)

            # Get starters efficiently
            starter_ids = {
                p["id"]
                for p in event_squad
                if solver.Value(starter_vars[(p["id"], event_id)]) == 1
            }

            starters = [p for p in event_squad if p["id"] in starter_ids]
            starters_by_event[event_id] = starters

        # Process transfers and penalties
        total_transfers = 0
        transfer_penalties = {}

        if current_squad and hasattr(self, "_event_transfer_vars_created"):
            # Process transfers for all events
            for event_id in event_ids:
                transfers_in = [
                    self._process_player_for_output(p)
                    for p in self.players
                    if solver.Value(self.event_transfer_in_vars[(p.id, event_id)]) == 1
                ]

                transfers_out = [
                    self._process_player_for_output(p)
                    for p in self.players
                    if solver.Value(self.event_transfer_out_vars[(p.id, event_id)]) == 1
                ]

                transfers_by_event[event_id] = {
                    "in": transfers_in,
                    "out": transfers_out,
                    "count": len(transfers_in),
                }

                total_transfers += len(transfers_in)

            # Calculate transfer penalties by phase
            phases = defaultdict(list)
            for event_id in event_ids:
                phase = self._get_gameweek_for_event(event_id)
                phases[phase].append(event_id)

            for phase, phase_events in phases.items():
                phase_transfers = sum(
                    len(transfers_by_event[event_id]["in"]) for event_id in phase_events
                )

                excess_transfers = max(0, phase_transfers - self.free_transfers)
                penalty = excess_transfers * self.config.TRANSFER_PENALTY

                transfer_penalties[phase] = {
                    "transfers": phase_transfers,
                    "excess": excess_transfers,
                    "penalty": penalty,
                }

        # Calculate points
        raw_points = solver.ObjectiveValue() / 1000
        total_penalty = (
            sum(p["penalty"] for p in transfer_penalties.values())
            if transfer_penalties
            else 0
        )
        adjusted_points = raw_points - total_penalty

        return {
            "squad": selected_players,
            "daily_starters": starters_by_event,
            "transfers_by_event": transfers_by_event,
            "transfer_summary": {
                "total_transfers": total_transfers,
                "penalties_by_phase": transfer_penalties,
                "total_penalty": total_penalty,
            },
            "points": {
                "raw_points": raw_points,
                "transfer_penalty": total_penalty,
                "adjusted_points": adjusted_points,
            },
            "total_cost": sum(p["cost"] for p in selected_players),
            "average_points_per_day": (
                adjusted_points / len(event_ids) if event_ids else 0
            ),
            "total_games": sum(
                len(starters) for starters in starters_by_event.values()
            ),
        }

    def _fallback_solution(
        self, event_ids: List[int], current_squad: List[Dict] = None
    ) -> Dict:
        """
        Simple greedy fallback solution if CP-SAT fails.
        """
        # Sort players by points per cost ratio
        scoring_metric = "form" if self.scoring_metric == "form" else "points_per_game"
        players_by_value = sorted(
            self.players,
            key=lambda p: getattr(p, scoring_metric) / p.now_cost,
            reverse=True,
        )

        # Select squad greedily while respecting constraints
        selected = []
        total_cost = 0
        team_counts = defaultdict(int)
        front_court = 0
        back_court = 0

        for player in players_by_value:
            if len(selected) >= self.config.SQUAD_SIZE:
                break

            cost = (
                player.selling_price
                if current_squad and player.selling_price
                else player.now_cost
            )

            if (
                total_cost + cost <= self.budget
                and team_counts[player.team] < self.config.MAX_PLAYERS_PER_TEAM
                and (
                    (
                        player.position == Position.FRONT_COURT
                        and front_court < self.config.FRONT_COURT_COUNT
                    )
                    or (
                        player.position == Position.BACK_COURT
                        and back_court < self.config.BACK_COURT_COUNT
                    )
                )
            ):
                selected.append(player)
                total_cost += cost
                team_counts[player.team] += 1
                if player.position == Position.FRONT_COURT:
                    front_court += 1
                else:
                    back_court += 1

        # Select starters greedily for each event
        starters_by_event = {}
        for event_id in event_ids:
            playing_teams = self._get_playing_teams(event_id)
            available_players = [p for p in selected if p.team in playing_teams]

            # Sort by points
            available_players.sort(
                key=lambda p: getattr(p, scoring_metric), reverse=True
            )

            # Select starters while respecting position constraints
            starters = []
            front_court = 0
            back_court = 0

            for player in available_players:
                if len(starters) >= self.config.STARTING_PLAYERS:
                    break

                if (player.position == Position.FRONT_COURT and front_court < 3) or (
                    player.position == Position.BACK_COURT and back_court < 3
                ):
                    starters.append(player)
                    if player.position == Position.FRONT_COURT:
                        front_court += 1
                    else:
                        back_court += 1

            starters_by_event[event_id] = starters

        # Calculate expected points
        expected_points = sum(
            sum(getattr(p, scoring_metric) for p in starters_by_event[event_id])
            for event_id in event_ids
        )

        return {
            "squad": selected,
            "daily_starters": starters_by_event,
            "transfers_by_gameweek": {},  # No transfers in fallback
            "expected_points": expected_points,
        }

    def _add_squad_evolution_constraints(self, model, sorted_events):
        """Add constraints for squad evolution across events"""
        if len(sorted_events) <= 1:
            return

        # For subsequent events, update squad based on previous event
        for i in range(1, len(sorted_events)):
            prev_event = sorted_events[i - 1]
            curr_event = sorted_events[i]

            for player in self.players:
                # Squad status update
                model.Add(
                    self.player_in_squad_vars[(player.id, curr_event)]
                    == self.player_in_squad_vars[(player.id, prev_event)]
                    + self.event_transfer_in_vars[(player.id, curr_event)]
                    - self.event_transfer_out_vars[(player.id, curr_event)]
                )

                # Transfer logic constraints
                model.Add(
                    self.event_transfer_in_vars[(player.id, curr_event)]
                    <= 1 - self.player_in_squad_vars[(player.id, prev_event)]
                )
                model.Add(
                    self.event_transfer_out_vars[(player.id, curr_event)]
                    <= self.player_in_squad_vars[(player.id, prev_event)]
                )

    def _add_final_squad_link_constraints(self, model, player_vars, sorted_events):
        """Link player_vars to final event's squad status"""
        if not sorted_events:
            return

        final_event = sorted_events[-1]
        for player in self.players:
            model.Add(
                player_vars[player.id]
                == self.player_in_squad_vars[(player.id, final_event)]
            )

    def _add_transfer_balance_constraints(self, model, sorted_events):
        """Ensure transfers in = transfers out for each event"""
        for event_id in sorted_events:
            model.Add(
                sum(self.event_transfer_in_vars[(p.id, event_id)] for p in self.players)
                == sum(
                    self.event_transfer_out_vars[(p.id, event_id)] for p in self.players
                )
            )

    def _add_position_balance_constraints(self, model, sorted_events):
        """Ensure position balance is maintained for each event"""
        # Pre-compute player lists by position for faster access
        front_court_players = self.players_by_position[Position.FRONT_COURT]
        back_court_players = self.players_by_position[Position.BACK_COURT]

        for event_id in sorted_events:
            # Front court count constraint
            front_court_vars = [
                self.player_in_squad_vars[(p.id, event_id)] for p in front_court_players
            ]
            model.Add(sum(front_court_vars) == self.config.FRONT_COURT_COUNT)

            # Back court count constraint
            back_court_vars = [
                self.player_in_squad_vars[(p.id, event_id)] for p in back_court_players
            ]
            model.Add(sum(back_court_vars) == self.config.BACK_COURT_COUNT)

            # For transfers specifically, ensure position balance
            if event_id != sorted_events[0]:
                # Front court transfers in = front court transfers out
                front_in_vars = [
                    self.event_transfer_in_vars[(p.id, event_id)]
                    for p in front_court_players
                ]
                front_out_vars = [
                    self.event_transfer_out_vars[(p.id, event_id)]
                    for p in front_court_players
                ]
                model.Add(sum(front_in_vars) == sum(front_out_vars))

                # Back court transfers in = back court transfers out
                back_in_vars = [
                    self.event_transfer_in_vars[(p.id, event_id)]
                    for p in back_court_players
                ]
                back_out_vars = [
                    self.event_transfer_out_vars[(p.id, event_id)]
                    for p in back_court_players
                ]
                model.Add(sum(back_in_vars) == sum(back_out_vars))

    def _add_budget_constraints(self, model, sorted_events):
        """Add budget constraints for each event"""
        for event_id in sorted_events:
            event_squad_cost = model.NewIntVar(
                0,
                self.budget * 10,  # Multiply by 10 for safety margin
                f"event_squad_cost_{event_id}",
            )

            model.Add(
                event_squad_cost
                == sum(
                    self.player_in_squad_vars[(p.id, event_id)] * p.get_cost()
                    for p in self.players
                )
            )

            model.Add(event_squad_cost <= self.budget)

    def _add_transfer_penalties(self, model, phases):
        """Add transfer penalties to the objective function"""
        objective_terms = []

        # Track transfers at phase level for penalty calculation
        for phase, phase_events in phases.items():
            phase_transfers_in = []

            # Collect all transfers for the phase
            for event_id in phase_events:
                phase_transfers_in.extend(
                    [
                        self.event_transfer_in_vars[(p.id, event_id)]
                        for p in self.players
                    ]
                )

            # Sum up transfers for the phase
            phase_transfer_sum = model.NewIntVar(
                0,
                len(self.players) * len(phase_events),
                f"phase_transfer_sum_{phase}",
            )
            model.Add(phase_transfer_sum == sum(phase_transfers_in))

            # Calculate excess transfers for the phase
            phase_excess_transfers = model.NewIntVar(
                0,
                len(self.players) * len(phase_events),
                f"phase_excess_transfers_{phase}",
            )
            model.Add(
                phase_excess_transfers
                >= phase_transfer_sum - self.free_transfers
            )
            model.Add(phase_excess_transfers >= 0)

            # Add transfer penalty to objective - scale it the same way as points
            objective_terms.append(
                -phase_excess_transfers
                * self._scale_points(self.config.TRANSFER_PENALTY)
            )

        return objective_terms
