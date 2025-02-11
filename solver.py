from collections import defaultdict
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from ortools.sat.python import cp_model
from typing import List, Dict, Set, Tuple


class Position(Enum):
    BACK_COURT = 1
    FRONT_COURT = 2

    @classmethod
    def from_element_type(cls, element_type: int) -> "Position":
        """Convert API element_type to Position enum."""
        return cls.BACK_COURT if element_type == 1 else cls.FRONT_COURT

    def __str__(self) -> str:
        return self.name


@dataclass
class Player:
    id: int
    name: str
    position: Position
    team: int
    form: float
    points_per_game: float
    now_cost: int
    selling_price: int = None


@dataclass
class Fixture:
    event: int
    team_h: int
    team_a: int


@dataclass
class Phase:
    id: int
    start_event: int
    stop_event: int


class TeamOptimizer:
    def __init__(
        self,
        players: List[Dict],
        fixtures: List[Dict],
        phases: List[Dict],
        teams: List[Dict],
        current_squad: List[Dict] = None,
        scoring_metric: str = "form",  # or "points_per_game"
    ):
        self.players = self._process_players(players, current_squad)
        self.fixtures = [
            Fixture(
                **{
                    k: v
                    for k, v in f.items()
                    if k in {field.name for field in fields(Fixture)}
                }
            )
            for f in fixtures
        ]  # Filter out unnecessary fields
        self.phases = [
            Phase(
                **{
                    k: v
                    for k, v in p.items()
                    if k in {field.name for field in fields(Phase)}
                }
            )
            for p in phases
        ]
        self.scoring_metric = scoring_metric
        self.BUDGET = (
            sum(current_squad["selling_price"] for player in current_squad)
            if current_squad
            else 1000
        )
        print("budget :>>", self.BUDGET)
        self.SQUAD_SIZE = 10
        self.FRONT_COURT_COUNT = 5
        self.BACK_COURT_COUNT = 5
        self.STARTING_PLAYERS = 5
        self.MAX_PLAYERS_PER_TEAM = 2
        self.FREE_TRANSFERS = 2
        self.TRANSFER_PENALTY = 1000
        self.team_lookup = {t["id"]: t for t in teams}
        self.team_names = {t["id"]: t["name"] for t in teams}
        self.team_short_names = {t["id"]: t["short_name"] for t in teams}

    def _process_players(
        self, players: List[Dict], current_squad: List[Dict] = None
    ) -> List[Player]:
        """Process raw player data into Player objects."""
        current_squad_map = {}
        if current_squad:
            current_squad_map = {
                p["element"]: p["selling_price"] for p in current_squad
            }

        processed_players = []
        for p in players:
            # Determine position based on element_type
            position = Position.from_element_type(p["element_type"])

            player = Player(
                id=p["id"],
                name=p["web_name"],
                position=position,
                team=p["team"],
                form=float(p["form"] or 0),
                points_per_game=float(p["points_per_game"] or 0),
                now_cost=p["now_cost"],
                selling_price=current_squad_map.get(p["id"]),
            )
            processed_players.append(player)

        return processed_players

    def _get_gameweek_for_event(self, event_id: int) -> int:
        """Get the gameweek (phase) ID for a given event ID."""
        for phase in self.phases:
            if phase.start_event <= event_id <= phase.stop_event:
                return phase.id
        return None

    def _get_playing_teams(self, event_id: int) -> Set[int]:
        """Get set of teams playing in a given event."""
        teams = set()
        for fixture in self.fixtures:
            if fixture.event == event_id:
                teams.add(fixture.team_h)
                teams.add(fixture.team_a)
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

    def optimize(self, event_ids: List[int], current_squad: List[Dict] = None) -> Dict:
        """
        Main optimization function that creates and solves the CP-SAT problem.
        Returns optimal squad and transfers for given event range.
        """
        model = cp_model.CpModel()

        # Create decision variables
        player_vars = {}  # Boolean vars for squad selection
        starter_vars = {}  # Boolean vars for starting lineup each event
        transfer_vars = {}  # Boolean vars for transfers

        for player in self.players:
            # Squad selection vars
            player_vars[player.id] = model.NewBoolVar(f"player_{player.id}")

            # Starting lineup vars for each event
            for event_id in event_ids:
                starter_vars[(player.id, event_id)] = model.NewBoolVar(
                    f"starter_{player.id}_{event_id}"
                )

            # Transfer vars for each gameweek
            gameweeks = set(self._get_gameweek_for_event(e) for e in event_ids)
            for gw in gameweeks:
                transfer_vars[(player.id, gw)] = model.NewBoolVar(
                    f"transfer_{player.id}_{gw}"
                )

        # Objective terms
        objective_terms = []

        # Points from starters
        for event_id in event_ids:
            playing_teams = self._get_playing_teams(event_id)
            for player in self.players:
                if player.team in playing_teams:
                    points = (
                        player.form
                        if self.scoring_metric == "form"
                        else player.points_per_game
                    )
                    objective_terms.append(
                        starter_vars[(player.id, event_id)]
                        * self._scale_points(float(points))
                    )

        # Transfer penalties
        gameweeks = set(self._get_gameweek_for_event(e) for e in event_ids)
        for gw in gameweeks:
            # Count transfers
            transfers = []
            for player in self.players:
                transfers.append(transfer_vars[(player.id, gw)])

            # Create penalty variable
            excess_transfers = model.NewIntVar(
                0, len(self.players), f"excess_transfers_{gw}"
            )

            # Set up BoolVars
            transfer_sum = model.NewIntVar(0, len(transfers), "transfer_sum")
            exceeds_free_transfers = model.NewBoolVar("exceeds_free_transfers")
            model.Add(transfer_sum > self.FREE_TRANSFERS).OnlyEnforceIf(
                exceeds_free_transfers
            )
            model.Add(transfer_sum <= self.FREE_TRANSFERS).OnlyEnforceIf(
                exceeds_free_transfers.Not()
            )

            # Sum transfers and subtract free transfers
            model.Add(excess_transfers >= transfer_sum - self.FREE_TRANSFERS)
            model.Add(
                excess_transfers <= transfer_sum - self.FREE_TRANSFERS
            ).OnlyEnforceIf(exceeds_free_transfers)
            model.Add(excess_transfers == 0).OnlyEnforceIf(exceeds_free_transfers.Not())

            # Add penalty to the objective
            objective_terms.append(-excess_transfers * self.TRANSFER_PENALTY)

        # Set objective
        model.Maximize(sum(objective_terms))

        # Constraints

        # 1. Squad size and position constraints
        model.Add(sum(player_vars[p.id] for p in self.players) == self.SQUAD_SIZE)
        model.Add(
            sum(
                player_vars[p.id]
                for p in self.players
                if p.position == Position.FRONT_COURT
            )
            == self.FRONT_COURT_COUNT
        )
        model.Add(
            sum(
                player_vars[p.id]
                for p in self.players
                if p.position == Position.BACK_COURT
            )
            == self.BACK_COURT_COUNT
        )

        # 2. Budget constraint
        min_squad_cost = sum(sorted([p.now_cost for p in self.players])[:10])
        if current_squad:
            model.Add(
                sum(
                    player_vars[p.id]
                    * (p.selling_price if p.selling_price else p.now_cost)
                    for p in self.players
                )
                <= self.BUDGET
            )
        else:
            model.Add(
                sum(player_vars[p.id] * p.now_cost for p in self.players) <= self.BUDGET
            )

        # 3. Team limit constraint
        for team_id in set(p.team for p in self.players):
            model.Add(
                sum(player_vars[p.id] for p in self.players if p.team == team_id)
                <= self.MAX_PLAYERS_PER_TEAM
            )

        # 4. Starting lineup constraints
        for event_id in event_ids:
            # Up to 5 starters per event
            eligible_players = sum(1 for p in self.players if p.team in playing_teams)
            model.Add(
                sum(starter_vars[(p.id, event_id)] for p in self.players)
                <= self.STARTING_PLAYERS
            )

            # Position balance in starting lineup (2-3 Front Court players)
            front_court_starters = sum(
                starter_vars[(p.id, event_id)]
                for p in self.players
                if p.position == Position.FRONT_COURT
            )
            back_court_starters = sum(
                starter_vars[(p.id, event_id)]
                for p in self.players
                if p.position == Position.BACK_COURT
            )
            model.Add(front_court_starters + back_court_starters <= 5)  # Total starters
            model.Add(front_court_starters <= 3)  # Max front court
            model.Add(back_court_starters <= 3)  # Max back court

            # Can only start players in squad
            for player in self.players:
                model.Add(starter_vars[(player.id, event_id)] <= player_vars[player.id])

            # Can only start players with games
            playing_teams = self._get_playing_teams(event_id)
            for player in self.players:
                if player.team not in playing_teams:
                    model.Add(starter_vars[(player.id, event_id)] == 0)

        # 5. Transfer constraints
        if current_squad:
            current_players = {p["element"] for p in current_squad}
            for gw in gameweeks:
                for player in self.players:
                    if player.id not in current_players:
                        # Transfer in: player wasn't in previous squad but is in new squad
                        model.Add(
                            transfer_vars[(player.id, gw)] >= player_vars[player.id]
                        )
                    else:
                        # Transfer out: player was in previous squad but isn't in new squad
                        model.Add(
                            transfer_vars[(player.id, gw)]
                            >= (1 - player_vars[player.id])
                        )

        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300  # 5 minute timeout
        status = solver.Solve(model)

        print(
            "Statuses :>>",
            cp_model.UNKNOWN,
            cp_model.MODEL_INVALID,
            cp_model.FEASIBLE,
            cp_model.INFEASIBLE,
            cp_model.OPTIMAL,
        )
        print(f"Solver status :>> {status}")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._fallback_solution(event_ids, current_squad)

        # Process results
        selected_players = [
            self._process_player_for_output(p)
            for p in self.players
            if solver.Value(player_vars[p.id]) == 1
        ]

        starters_by_event = defaultdict(list)
        for event_id in event_ids:
            starters_by_event[event_id] = [
                p
                for p in selected_players
                if solver.Value(starter_vars[(p["id"], event_id)]) == 1
            ]

        transfers_by_gw = defaultdict(list)
        if current_squad:
            for gw in gameweeks:
                transfers_by_gw[gw] = [
                    p
                    for p in self.players
                    if solver.Value(transfer_vars[(p.id, gw)]) == 1
                ]

        total_cost = sum(p["cost"] for p in selected_players)
        total_points = solver.ObjectiveValue() / 1000
        average_points = total_points / len(event_ids) if event_ids else 0
        total_games = sum(len(starters) for starters in starters_by_event.values())

        return {
            "squad": selected_players,
            "daily_starters": dict(starters_by_event),
            "transfers_by_gameweek": dict(transfers_by_gw),
            "expected_points": total_points,
            "total_cost": total_cost,
            "average_points_per_day": average_points,
            "total_games": total_games,
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
            if len(selected) >= self.SQUAD_SIZE:
                break

            cost = (
                player.selling_price
                if current_squad and player.selling_price
                else player.now_cost
            )

            if (
                total_cost + cost <= self.BUDGET
                and team_counts[player.team] < self.MAX_PLAYERS_PER_TEAM
                and (
                    (
                        player.position == Position.FRONT_COURT
                        and front_court < self.FRONT_COURT_COUNT
                    )
                    or (
                        player.position == Position.BACK_COURT
                        and back_court < self.BACK_COURT_COUNT
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
                if len(starters) >= self.STARTING_PLAYERS:
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
