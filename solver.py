from collections import defaultdict
from enum import Enum
from ortools.linear_solver import pywraplp
from typing import List, Dict, Any, Tuple


class Position(Enum):
    BACK_COURT = 1
    FRONT_COURT = 2


class TeamOptimizer:
    def __init__(self, data: Dict[str, Any], fixtures: List[Any]):
        """
        Initialize the optimizer with game data
        """
        self.data = data
        self.fixtures = fixtures
        self.team_map = {team["id"]: team["short_name"] for team in data["teams"]}
        self.player_map = {p["id"]: p for p in data["elements"]}

    def get_player_games(self, gamedays: List[int]) -> Dict[int, List[str]]:
        """
        Compute games distribution per player for given gamedays
        """
        player_games = defaultdict(list)
        for fixture in self.fixtures:
            if fixture["event"] in gamedays:
                game_date = fixture["kickoff_time"][:10]
                home_team_players = [
                    p["id"]
                    for p in self.data["elements"]
                    if p["team"] == fixture["team_h"]
                ]
                away_team_players = [
                    p["id"]
                    for p in self.data["elements"]
                    if p["team"] == fixture["team_a"]
                ]

                for player_id in home_team_players + away_team_players:
                    player_games[player_id].append(game_date)
        return player_games

    def optimize(
        self,
        gamedays: List[int],
        budget: int = 1000,
        num_players: int = 10,
        max_per_team: int = 2,
        num_front_court: int = 5,
        num_back_court: int = 5,
        max_daily_scorers: int = 5,
        min_pos_scorers: int = 2,
        max_pos_scorers: int = 3,
        points_column: str = "form"
    ) -> Dict:
        """
        Optimize team selection and daily lineups with position-balanced scoring
        """
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise Exception("Solver not available")

        # Get player game distribution
        player_games = self.get_player_games(gamedays)
        unique_gamedays = sorted(
            set(day for games in player_games.values() for day in games)
        )

        # Decision Variables
        player_vars = {}  # Squad selection
        scorer_vars = {}  # Whether player scores on a given day
        has_scorers = {}  # Whether we have any scorers on a given day

        # Create variables
        for player in self.data["elements"]:
            player_id = player["id"]
            player_vars[player_id] = solver.BoolVar(f"squad_{player_id}")
            scorer_vars[player_id] = {}
            for gameday in unique_gamedays:
                if gameday in player_games[player_id]:
                    scorer_vars[player_id][gameday] = solver.BoolVar(
                        f"scorer_{player_id}_{gameday}"
                    )

        # Create variables for days with scorers
        for gameday in unique_gamedays:
            has_scorers[gameday] = solver.BoolVar(f"has_scorers_{gameday}")

        # Squad Constraints

        # 1. Total players constraint
        solver.Add(solver.Sum(player_vars.values()) == num_players)

        # 2. Budget constraint
        solver.Add(
            solver.Sum(
                player_vars[p] * self.player_map[p]["now_cost"] for p in player_vars
            )
            <= budget
        )

        # 3. Position constraints in squad
        solver.Add(
            solver.Sum(
                player_vars[p]
                for p in player_vars
                if self.player_map[p]["element_type"] == Position.FRONT_COURT.value
            )
            == num_front_court
        )
        solver.Add(
            solver.Sum(
                player_vars[p]
                for p in player_vars
                if self.player_map[p]["element_type"] == Position.BACK_COURT.value
            )
            == num_back_court
        )

        # 4. Team limit constraint
        for team_id in self.team_map:
            solver.Add(
                solver.Sum(
                    player_vars[p]
                    for p in player_vars
                    if self.player_map[p]["team"] == team_id
                )
                <= max_per_team
            )

        # Daily Scoring Constraints
        for gameday in unique_gamedays:
            playing_today = [p for p in player_vars if gameday in player_games[p]]

            if playing_today:
                # 5. Can only score if in squad and playing
                for player_id in playing_today:
                    solver.Add(
                        scorer_vars[player_id][gameday] <= player_vars[player_id]
                    )

                # Get players by position
                front_court_today = [
                    p
                    for p in playing_today
                    if self.player_map[p]["element_type"] == Position.FRONT_COURT
                ]
                back_court_today = [
                    p
                    for p in playing_today
                    if self.player_map[p]["element_type"] == Position.BACK_COURT
                ]

                # 6. Link has_scorers to whether we use any players that day
                solver.Add(
                    solver.Sum(scorer_vars[p][gameday] for p in playing_today)
                    <= max_daily_scorers * has_scorers[gameday]
                )
                solver.Add(
                    solver.Sum(scorer_vars[p][gameday] for p in playing_today)
                    >= has_scorers[gameday]
                )

                if front_court_today and back_court_today:
                    # 7. Position balance constraints when we have scorers
                    # Front court constraints
                    solver.Add(
                        solver.Sum(scorer_vars[p][gameday] for p in front_court_today)
                        >= min_pos_scorers * has_scorers[gameday]
                    )
                    solver.Add(
                        solver.Sum(scorer_vars[p][gameday] for p in front_court_today)
                        <= max_pos_scorers
                    )

                    # Back court constraints
                    solver.Add(
                        solver.Sum(scorer_vars[p][gameday] for p in back_court_today)
                        >= min_pos_scorers * has_scorers[gameday]
                    )
                    solver.Add(
                        solver.Sum(scorer_vars[p][gameday] for p in back_court_today)
                        <= max_pos_scorers
                    )

        # Objective: Maximize total points from scoring players
        objective = solver.Objective()
        for gameday in unique_gamedays:
            playing_today = [p for p in player_vars if gameday in player_games[p]]
            for player_id in playing_today:
                points = float(self.player_map[player_id][points_column])
                objective.SetCoefficient(scorer_vars[player_id][gameday], points)
        objective.SetMaximization()

        # Solve
        status = solver.Solve()
        print(f"status: {status}")

        if status == pywraplp.Solver.OPTIMAL:
            # Extract solution
            selected_players = []
            for p in player_vars:
                if player_vars[p].solution_value() > 0.5:
                    player_info = {
                        "name": self.player_map[p]["web_name"],
                        "team": self.team_map[self.player_map[p]["team"]],
                        "games": len(player_games[p]),
                        "points_per_game": float(self.player_map[p][points_column]),
                        "position": (
                            "Front Court"
                            if self.player_map[p]["element_type"] == Position.FRONT_COURT.value
                            else "Back Court"
                        ),
                        "cost": self.player_map[p]["now_cost"],
                    }
                    selected_players.append(player_info)

            daily_scorers = {}
            for gameday in unique_gamedays:
                if has_scorers[gameday].solution_value() > 0.5:
                    scorers = [
                        {
                            "name": self.player_map[p]["web_name"],
                            "team": self.team_map[self.player_map[p]["team"]],
                            "points": float(self.player_map[p][points_column]),
                            "position": (
                                "Front Court"
                                if self.player_map[p]["element_type"] == Position.FRONT_COURT.value
                                else "Back Court"
                            ),
                        }
                        for p in player_vars
                        if gameday in scorer_vars[p]
                        and scorer_vars[p][gameday].solution_value() > 0.5
                    ]
                    if scorers:
                        daily_scorers[gameday] = scorers

            total_points = solver.Objective().Value()
            total_cost = sum(
                self.player_map[p]["now_cost"]
                for p in player_vars
                if player_vars[p].solution_value() > 0.5
            )

            return {
                "selected_players": selected_players,
                "daily_scorers": daily_scorers,
                "total_points": total_points,
                "total_cost": total_cost,
                "average_points_per_day": total_points / len(unique_gamedays),
                "scoring_days": len(daily_scorers),
            }
        else:
            raise ValueError("No optimal solution found! Check constraints.")
