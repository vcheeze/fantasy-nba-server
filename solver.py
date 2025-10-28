from collections import defaultdict
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from ortools.sat.python import cp_model
import os
from typing import NamedTuple, Optional


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
        players: list[dict],
        fixtures: list[dict],
        phases: list[dict],
        teams: list[dict],
        scoring_metric: str = "form",
        current_squad: Optional[list[dict]] = None,
        config: Optional[OptimizationConfig] = None,
        transfers: Optional[Transfers] = None,
    ):
        print(f"transfers :>> {transfers}")
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
        # Create a player_map for fast ID lookups
        self.player_map = {p.id: p for p in self.players}

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
        self, players: list[dict], current_squad: Optional[list[dict]]
    ) -> list[Player]:
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

    def _process_fixtures(self, fixtures: list[dict]) -> list[Fixture]:
        """Process fixtures with field filtering."""
        fixture_fields = {field.name for field in fields(Fixture)}
        return [
            Fixture(**{k: v for k, v in f.items() if k in fixture_fields})
            for f in fixtures
        ]

    def _process_phases(self, phases: list[dict]) -> list[Phase]:
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
        self, event_ids: list[int]
    ) -> tuple[cp_model.CpModel, dict, dict]:
        """Create CP-SAT model and variables."""
        model = cp_model.CpModel()

        player_vars = {}  # Boolean vars for FINAL squad selection
        transfer_vars = {}  # Boolean vars for transfers (per gameweek for penalty)

        # --- Multi-Stage Roster Variables ---
        self.player_in_squad_vars = {}
        self.event_transfer_in_vars = {}
        self.event_transfer_out_vars = {}

        # Batch variable creation
        sorted_events = sorted(list(event_ids))
        gameweeks = {self._get_gameweek_for_event(e) for e in event_ids}

        for player in self.players:
            # Final squad selection var
            player_vars[player.id] = model.NewBoolVar(f"player_{player.id}")

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

            # Transfer vars for each gameweek (for penalty)
            for gw in gameweeks:
                if gw is not None:  # Skip None values
                    transfer_vars[(player.id, gw)] = model.NewBoolVar(
                        f"transfer_{player.id}_{gw}"
                    )

        # Add symmetry breaking constraints for similar players
        self._add_symmetry_breaking(model, player_vars)

        # Return model and only the relevant variable dictionaries
        return model, player_vars, transfer_vars

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
            if phase:
                phases[phase].append(event_id)

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
                self.budget * 10,  # Use a reasonable upper bound
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
        self._event_transfer_vars_created = True  # Flag for solution processing

        # Track transfers at phase level for penalty calculation
        for phase, phase_events in phases.items():
            if not phase_events:
                continue

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

            # Use AddMaxEquality for max(0, sum - free)
            model.AddMaxEquality(
                phase_excess_transfers, [phase_transfer_sum - self.free_transfers, 0]
            )

            # Add transfer penalty to objective - scale it the same way as points
            objective_terms.append(
                -phase_excess_transfers
                * self._scale_points(self.config.TRANSFER_PENALTY)
            )

        return objective_terms

    def _add_objective_terms(
        self, model: cp_model.CpModel, event_ids: list[int]
    ) -> list:
        """
        Adds the objective terms to the model.
        This implementation uses the 10-man roster score as a proxy.
        """
        objective_terms = []

        for event_id in event_ids:
            playing_teams = self._get_playing_teams(event_id)

            for player in self.players:
                if player.team in playing_teams:
                    points = player.get_score(self.scoring_metric)
                    scaled_points = self._scale_points(points)

                    # Add points to objective if player is in the squad for this event
                    objective_terms.append(
                        self.player_in_squad_vars[(player.id, event_id)] * scaled_points
                    )

        return objective_terms

    def _add_constraints(
        self,
        model: cp_model.CpModel,
        player_vars: dict,
        transfer_vars: dict,  # This variable is not actually used, but kept for signature consistency
        event_ids: list[int],
        current_squad: Optional[list[dict]],
        wildcard: Optional[bool] = False,
    ) -> list:
        """Add constraints to the optimization model."""
        objective_terms = []
        sorted_events = sorted(event_ids)

        # ===== 1. SQUAD COMPOSITION CONSTRAINTS (for FINAL squad) =====
        # These constraints are applied to player_vars, which is the *final* roster.
        # The per-event constraints are handled in the evolution/transfer logic.
        model.Add(sum(player_vars.values()) == self.config.SQUAD_SIZE)

        for position, players in self.players_by_position.items():
            count = (
                self.config.FRONT_COURT_COUNT
                if position == Position.FRONT_COURT
                else self.config.BACK_COURT_COUNT
            )
            model.Add(sum(player_vars[p.id] for p in players) == count)

        model.Add(
            sum(player_vars[p.id] * p.get_cost() for p in self.players) <= self.budget
        )

        for team_players in self.players_by_team.values():
            model.Add(
                sum(player_vars[p.id] for p in team_players)
                <= self.config.MAX_PLAYERS_PER_TEAM
            )

        # ===== 2. TRANSFER/SQUAD EVOLUTION CONSTRAINTS (if applicable) =====
        if current_squad and not wildcard:
            objective_terms.extend(
                self._add_transfer_constraints(
                    model, player_vars, event_ids, sorted_events, current_squad
                )
            )
        else:
            # No current squad (build from scratch) or wildcard
            # We still need to set up the evolution and link the final squad
            self._add_squad_evolution_constraints(model, sorted_events)
            self._add_final_squad_link_constraints(model, player_vars, sorted_events)

            # Apply per-event constraints
            self._add_transfer_balance_constraints(model, sorted_events)
            self._add_position_balance_constraints(model, sorted_events)
            self._add_budget_constraints(model, sorted_events)

            # Group events by phase for penalty calculation
            phases = defaultdict(list)
            for event_id in event_ids:
                phase = self._get_gameweek_for_event(event_id)
                if phase:
                    phases[phase].append(event_id)
            objective_terms.extend(self._add_transfer_penalties(model, phases))

        # ===== 3. OBJECTIVE FUNCTION TERMS (based on roster, not starters) =====
        objective_terms.extend(self._add_objective_terms(model, event_ids))

        return objective_terms

    def optimize(
        self,
        event_ids: list[int],
        current_squad: Optional[list[dict]] = None,
        wildcard: Optional[bool] = False,
    ) -> dict:
        """
        Optimize team selection for given events with performance optimizations.
        """
        # Early validation
        if not event_ids:
            return {"error": "No events provided for optimization"}

        try:
            # Create model and variables
            print("Creating optimization model...")
            model, player_vars, transfer_vars = self._create_optimization_model(
                event_ids
            )

            # Add constraints and get objective terms
            print("Adding constraints and objective terms to the model...")
            objective_terms = self._add_constraints(
                model,
                player_vars,
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
            self._add_solver_hints(model, player_vars, event_ids)

            # Solve the model
            print("Solving...")
            status = solver.Solve(model)

            print(f"Solver status :>> {status} ({solver.StatusName()})")

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                return self._process_solution(
                    solver,
                    player_vars,
                    transfer_vars,
                    event_ids,
                    current_squad,
                )
            else:
                return {
                    "status": "Infeasible",
                    "solver": "cpsat",
                    "error": "No solution found by the solver.",
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

    def _add_solver_hints(self, model, player_vars, event_ids):
        """Add hints to guide the solver toward good solutions."""
        # Track which players we've already added hints for
        hinted_players = set()

        # Hint: Start with high-value players
        for player in sorted(
            self.players,
            key=lambda p: (
                p.get_score(self.scoring_metric) / p.get_cost()
                if p.get_cost() > 0
                else 0
            ),
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
                for player in self.players_by_team.get(team_id, [])[
                    :5
                ]:  # Top 5 players per team
                    if player.id not in hinted_players:
                        model.AddHint(player_vars[player.id], 1)
                        hinted_players.add(player.id)

    def _calculate_true_gameday_score(
        self, roster_player_ids: list[int], event_id: int
    ) -> tuple[float, list[dict]]:
        """
        Calculates the *true* maximized score for a given 10-player roster
        by finding the optimal 3-2 or 2-3 starting lineup for the gameday.
        """
        playing_type_1 = []  # Back Court
        playing_type_2 = []  # Front Court

        playing_teams = self._get_playing_teams(event_id)

        for p_id in roster_player_ids:
            # Use the fast player_map lookup
            player = self.player_map.get(p_id)
            if not player:
                continue

            if player.team in playing_teams:
                # Use 'points_per_game' for actual scoring, not the optimization metric
                ppg = player.points_per_game

                player_data = {
                    "id": player.id,
                    "name": player.name,
                    "team": self.team_short_names.get(player.team),
                    "position": player.position,
                    "points": ppg,
                }

                if player.position == Position.BACK_COURT:
                    playing_type_1.append(player_data)
                else:
                    playing_type_2.append(player_data)

        # Sort by points_per_game to easily find the top N
        playing_type_1.sort(key=lambda x: x["points"], reverse=True)
        playing_type_2.sort(key=lambda x: x["points"], reverse=True)

        # Case 1: 3 Back Court (Type 1), 2 Front Court (Type 2)
        score_3_2 = sum(p["points"] for p in playing_type_1[:3]) + sum(
            p["points"] for p in playing_type_2[:2]
        )
        starters_3_2 = playing_type_1[:3] + playing_type_2[:2]

        # Case 2: 2 Back Court (Type 1), 3 Front Court (Type 2)
        score_2_3 = sum(p["points"] for p in playing_type_1[:2]) + sum(
            p["points"] for p in playing_type_2[:3]
        )
        starters_2_3 = playing_type_1[:2] + playing_type_2[:3]

        if score_3_2 >= score_2_3:
            return score_3_2, starters_3_2
        else:
            return score_2_3, starters_2_3

    def _process_solution(
        self,
        solver: cp_model.CpSolver,
        player_vars: dict,
        transfer_vars: dict,
        event_ids: list[int],
        current_squad: Optional[list[dict]],
    ) -> dict:
        """Process optimization solution into result format."""

        # Get selected players for the *final* squad
        final_player_ids = {
            p_id for p_id, var in player_vars.items() if solver.Value(var) == 1
        }
        final_squad_players = [
            self._process_player_for_output(self.player_map[p_id])
            for p_id in final_player_ids
            if p_id in self.player_map
        ]

        # --- Process results for all events at once ---
        daily_starters = {}
        all_gameday_rosters = {}
        total_true_score = 0

        sorted_events = sorted(list(event_ids))

        # Check if the multi-stage variables were created
        has_event_vars = (
            hasattr(self, "_event_transfer_vars_created")
            and self._event_transfer_vars_created
        )

        if has_event_vars:
            for event_id in sorted_events:
                event_squad_ids = [
                    p.id
                    for p in self.players
                    if (p.id, event_id) in self.player_in_squad_vars
                    and solver.Value(self.player_in_squad_vars[(p.id, event_id)]) == 1
                ]
                all_gameday_rosters[event_id] = [
                    self._process_player_for_output(self.player_map[p_id])
                    for p_id in event_squad_ids
                    if p_id in self.player_map
                ]

                # Calculate true score for this event's roster
                true_score, starters = self._calculate_true_gameday_score(
                    event_squad_ids, event_id
                )
                total_true_score += true_score
                daily_starters[event_id] = {"score": true_score, "starters": starters}
        else:
            # This block might be hit if current_squad=None and wildcard=True (not fully modeled)
            # Fallback to assuming the final roster was used for all events
            for event_id in sorted_events:
                all_gameday_rosters[event_id] = final_squad_players
                true_score, starters = self._calculate_true_gameday_score(
                    final_player_ids, event_id
                )
                total_true_score += true_score
                daily_starters[event_id] = {"score": true_score, "starters": starters}

        # Collate transfers
        transfers_by_event = {
            event_id: {"in": [], "out": []} for event_id in sorted_events
        }
        paid_transfers_total = 0

        if has_event_vars:
            # Group events by phase
            phases = defaultdict(list)
            for event_id in event_ids:
                phase = self._get_gameweek_for_event(event_id)
                if phase:
                    phases[phase].append(event_id)

            for phase, phase_events in phases.items():
                phase_transfers_in_count = 0
                for event_id in phase_events:
                    for player in self.players:
                        if (
                            solver.Value(
                                self.event_transfer_in_vars[(player.id, event_id)]
                            )
                            == 1
                        ):
                            transfers_by_event[event_id]["in"].append(
                                {
                                    "phase": phase,
                                    "player": self._process_player_for_output(player),
                                }
                            )
                            phase_transfers_in_count += 1
                        if (
                            solver.Value(
                                self.event_transfer_out_vars[(player.id, event_id)]
                            )
                            == 1
                        ):
                            transfers_by_event[event_id]["out"].append(
                                {
                                    "phase": phase,
                                    "player": self._process_player_for_output(player),
                                }
                            )

                paid_transfers_total += max(
                    0, phase_transfers_in_count - self.free_transfers
                )

        elif current_squad:
            # Fallback for simple case: Compare initial and final
            current_squad_ids = {p["element"] for p in current_squad}
            t_out_ids = current_squad_ids - final_player_ids
            t_in_ids = final_player_ids - current_squad_ids

            # Attach those transfers to the first event (best-effort single-event placement)
            first_event = sorted_events[0] if sorted_events else None
            phase_for_event = (
                self._get_gameweek_for_event(first_event) if first_event else None
            )

            if first_event is not None:
                for p_id in t_in_ids:
                    if p_id in self.player_map:
                        transfers_by_event[first_event]["in"].append(
                            {
                                "phase": phase_for_event,
                                "player": self._process_player_for_output(
                                    self.player_map[p_id]
                                ),
                            }
                        )
                for p_id in t_out_ids:
                    if p_id in self.player_map:
                        transfers_by_event[first_event]["out"].append(
                            {
                                "phase": phase_for_event,
                                "player": self._process_player_for_output(
                                    self.player_map[p_id]
                                ),
                            }
                        )
            paid_transfers_total = max(0, len(t_in_ids) - self.free_transfers)

        # Filter out events that have no transfers at all
        transfers_by_event = {
            str(event_id): data
            for event_id, data in transfers_by_event.items()
            if data["in"] or data["out"]
        }
        total_transfers = sum(
            len(transfers["in"]) for transfers in transfers_by_event.values()
        )
        total_games = sum(len(daily["starters"]) for daily in daily_starters.values())
        average_points_per_day = total_true_score / len(sorted_events)

        return {
            "status": "Optimal" if solver.StatusName() == "OPTIMAL" else "Feasible",
            # "solver": "cpsat",
            "squad": final_squad_players,
            # "all_gameday_rosters": all_gameday_rosters,
            "total_cost": sum(p["cost"] for p in final_squad_players),
            "solver_objective_value": solver.ObjectiveValue() / 1000.0,  # Rescale
            "true_gameweek_score": total_true_score,
            "total_games": total_games,
            "daily_starters": daily_starters,
            "average_points_per_day": average_points_per_day,
            "transfers": {
                "by_event": transfers_by_event,
                "total": total_transfers,
                "paid": paid_transfers_total,
                "cost": paid_transfers_total * self.config.TRANSFER_PENALTY,
            },
        }
