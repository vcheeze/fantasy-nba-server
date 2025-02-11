[![example-fastapi](https://github.com/koyeb/example-fastapi/actions/workflows/deploy.yaml/badge.svg)](https://github.com/koyeb/example-fastapi/actions)

<div align="center">
  <a href="https://koyeb.com">
    <img src="https://www.koyeb.com/static/images/icons/koyeb.svg" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">Koyeb Serverless Platform</h3>
  <p align="center">
    Deploy a Python FastAPI application on Koyeb
    <br />
    <a href="https://koyeb.com">Learn more about Koyeb</a>
    ·
    <a href="https://koyeb.com/docs">Explore the documentation</a>
    ·
    <a href="https://koyeb.com/tutorials">Discover our tutorials</a>
  </p>
</div>

## About Koyeb and the Python FastAPI example application

Koyeb is a developer-friendly serverless platform to deploy apps globally. No-ops, servers, or infrastructure management.
This repository contains a Python FastAPI application you can deploy on the Koyeb serverless platform for testing.

This example application is designed to show how a Python FastAPI application can be deployed on Koyeb.

## Getting Started

Follow the steps below to deploy and run the Python FastAPI application on your Koyeb account.

### Requirements

You need a Koyeb account to successfully deploy and run this application. If you don't already have an account, you can sign-up for free [here](https://app.koyeb.com/auth/signup).

### Deploy using the Koyeb button

The fastest way to deploy the Python FastAPI application is to click the **Deploy to Koyeb** button below.

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&repository=github.com/koyeb/example-fastapi&branch=main&name=fastapi-on-koyeb)

Clicking on this button brings you to the Koyeb App creation page with everything pre-set to launch this application.

_To modify this application example, you will need to fork this repository. Checkout the [fork and deploy](#fork-and-deploy-to-koyeb) instructions._

### Fork and deploy to Koyeb

If you want to customize and enhance this application, you need to fork this repository.

If you used the **Deploy to Koyeb** button, you can simply link your service to your forked repository to be able to push changes.
Alternatively, you can manually create the application as described below.

On the [Koyeb Control Panel](https://app.koyeb.com/), on the **Overview** tab, click the **Create Web Service** button to begin.

1. Select **GitHub** as the deployment method.
2. In the repositories list, select the repository you just forked.
3. Choose a name for your App and Service, i.e `fastapi-on-koyeb`, and click **Deploy**.

You land on the deployment page where you can follow the build of your FastAPI application. Once the build has completed, your application is deployed and you will be able to access it via `<YOUR_APP_NAME>-<YOUR_ORG_NAME>.koyeb.app`.

## Contributing

If you have any questions, ideas or suggestions regarding this application sample, feel free to open an [issue](//github.com//koyeb/example-fastapi/issues) or fork this repository and open a [pull request](//github.com/koyeb/example-fastapi/pulls).

## Contact

[Koyeb](https://www.koyeb.com) - [@gokoyeb](https://twitter.com/gokoyeb) - [Slack](http://slack.koyeb.com/) - [Community](https://community.koyeb.com/)

## Prompt Template

# Fantasy NBA Salary Cap Edition Problem Statement

*Objective: Optimize team selection to yield the maximum total points while adhering to the following constraints:*

## Constraints:

1. Team Selection:
   - Budget of $1000
   - Should consist of exactly 10 players, with 5 Front Court and 5 Back Court players
   - No more than 2 players from the same team
   - 2 free transfers are allowed per Gameweek, which can be used separately on any given Gameday. Each additional transfer costs -1000 points
   - If the player is in the current squad, instead of using his `now_cost` to calculate against the total budget, we use the user's `selling_price` from the current squad API's `picks` field
2. Scoring
   - The season is divided into Gameweeks (phases), which are then divided into Gamedays (events)
   - Each Gameday, select exactly 5 players to start whose fantasy points count toward the total score
   - If more/fewer than 5 players have games on any given Gameday, they will score 0 points. Only the selected 5 starting players' points count toward the total score.
   - The 5 players have to consist of 2-3 Front Court and 2-3 Back Court players, and this formation cannot be broken

## Data

Player, team, and game data are fetched from an API, the models are as follows in JSON format. `events` refer to a Gameday, and `phases` refer to Gameweeks, which also detail what `events` are included in a given `phase`. `teams` refer to the NBA teams, while `elements` include all the players.

```json
{
    "events": [
        {
            "id": 1,
            "name": "Gameweek 1 - Day 1",
            "deadline_time": "2024-10-22T23:00:00Z",
            "release_time": null,
            "average_entry_score": 493,
            "finished": true,
            "data_checked": false,
            "highest_scoring_entry": 6096,
            "deadline_time_epoch": 1729638000,
            "deadline_time_game_offset": 0,
            "highest_score": 3007,
            "is_previous": false,
            "is_current": false,
            "is_next": false,
            "cup_leagues_created": false,
            "h2h_ko_matches_created": false,
            "ranked_count": 78437,
            "chip_plays": [
                {
                    "chip_name": "phcapt",
                    "num_played": 37904
                }
            ],
            "most_selected": 700,
            "most_transferred_in": 1,
            "top_element": 268,
            "top_element_info": {
                "id": 268,
                "points": 722
            },
            "transfers_made": 0,
            "most_captained": 157
        }
    ],
    "phases": [
        {
            "id": 1,
            "name": "Overall",
            "start_event": 1,
            "stop_event": 162,
            "highest_score": 161943
        },
        {
            "id": 2,
            "name": "Gameweek 1",
            "start_event": 1,
            "stop_event": 6,
            "highest_score": 13559
        }
    ],
    "teams": [
        {
            "code": 1610612737,
            "draw": 0,
            "form": null,
            "id": 1,
            "loss": 0,
            "name": "Atlanta Hawks",
            "played": 0,
            "points": 0,
            "position": 0,
            "short_name": "ATL",
            "strength": null,
            "team_division": null,
            "unavailable": false,
            "win": 0,
            "city": "Atlanta",
            "division": "Southeast",
            "conference": "East",
            "state": "GA"
        }
    ],
    "total_players": 127197,
    "elements": [
        {
            "chance_of_playing_next_round": 100,
            "chance_of_playing_this_round": 100,
            "code": 1629027,
            "cost_change_event": 0,
            "cost_change_event_fall": 0,
            "cost_change_start": 0,
            "cost_change_start_fall": 0,
            "dreamteam_count": 14,
            "element_type": 1,
            "ep_next": "0.0",
            "ep_this": "355.7",
            "event_points": 322,
            "first_name": "Trae",
            "form": "355.7",
            "id": 1,
            "in_dreamteam": false,
            "news": "",
            "news_added": "2025-01-29T22:15:04.405422Z",
            "now_cost": 160,
            "photo": "1629027.jpg",
            "points_per_game": "434.7",
            "second_name": "Young",
            "selected_by_percent": "2.2",
            "special": false,
            "squad_number": null,
            "status": "a",
            "team": 1,
            "team_code": 1610612737,
            "total_points": 19125,
            "transfers_in": 5012,
            "transfers_in_event": 0,
            "transfers_out": 4619,
            "transfers_out_event": 0,
            "value_form": "22.2",
            "value_season": "1195.3",
            "web_name": "T.Young",
            "region": null,
            "minutes": 1560,
            "points_scored": 992,
            "rebounds": 145,
            "assists": 501,
            "blocks": 10,
            "steals": 56,
            "turnovers": 203
        }
    ]
}
```

The fixtures are retrieved from another API, and they are linked with the `phases` field by their `event` field and a phase's `start_event` and `stop_event` fields. We can check if a player plays on any given Gameday by checking if their team is either `team_a` or `team_h`

```json
{
    "code": 22400754,
    "event": 106,
    "finished": false,
    "finished_provisional": false,
    "id": 754,
    "kickoff_time": "2025-02-09T19:00:00Z",
    "minutes": 0,
    "provisional_start_time": false,
    "started": false,
    "team_a": 19,
    "team_a_score": null,
    "team_h": 13,
    "team_h_score": null,
    "stats": [],
    "arena_name": "Fiserv Forum",
    "arena_city": "Milwaukee"
}
```

The current squad come from an API that contains the following information:

```json
{
  "picks": [
    {
      "element": 732,
      "position": 1,
      "selling_price": 58,
      "multiplier": 1,
      "purchase_price": 59,
      "is_captain": false
    },
    {
      "element": 588,
      "position": 2,
      "selling_price": 175,
      "multiplier": 2,
      "purchase_price": 173,
      "is_captain": true
    },
    // ... rest of the picks
  ],
  "transfers": {
    "cost": 1000,
    "status": "cost",
    "limit": 2,
    "made": 0,
    "bank": 1,
    "value": 1013
  }
}
```

## Goal

Your goal is to design a solution to this problem using LP or MIP to maximize the total points scored given any date range by utilizing the data available and adhering to the constraints given.

## Clarifying Questions

1. Optimization Scope:

- Question: Should the optimizer select a team for a single Gameweek, a fixed range of Gameweeks, or the entire season?
  Anser: An array of Gameday IDs will be passed as arguments, which will determine the range to optimize for
- Question: Do you want the optimizer to consider transfers across multiple Gameweeks dynamically?
  Answer: Yes, if the Gameday IDs passed in consist of multiple Gameweeks, take transfers as well as potential hits into account, as long as points as maximized

2. Scoring Mechanism Clarification:

- Question: The optimizer selects exactly 5 starters per Gameday. Does this mean unused players don’t contribute at all to scoring?
  Answer: Correct - unused players do not contribute to scoring even if they have games on any given day
- Question: If fewer than 5 players are available on a Gameday, does the optimizer attempt to spread out the games, or does it just maximize total points over the selected range?
  Answer: The tool needs to optimize overall points, which involves choosing players with the highest points based on their `form` or `points_per_game` dynamically (through function params), as well as spreading games out to ensure the most games are played by the players with the best form/points_per_game

3. Point Calculation for Transfers:

- Question: Are we penalizing additional transfers beyond the 2 free ones within a Gameweek with -100 points, as in your previous request?
  Answer: Yes, we penalize additional transfers with -1000 points
- Question: Should the optimizer attempt to preserve previous squad selections where beneficial (e.g., to avoid selling fees)?
  Answer: The optimizer should consider the best free transfers that will allow the highest gain in points, even if that involves transfer penalization. It could also involve performing no transfers if that yields more points

4. Solver Preference:

- Question: Are you open to using Google OR-Tools, PuLP (CBC solver), or a commercial solver like Gurobi?
  Answer: Only consider free solvers
- Question: Would you like a fallback method in case a solver doesn’t find an optimal solution?
  Answer: Yes, write a simple fallback if you can

5. Runtime Expectations:

- Question: Should the optimizer run quickly (seconds) for quick decision-making, or are longer runs (minutes) acceptable to get the best solution?
  Answer: Priotize getting maximum points, then account for performance and efficiency

---

# Current Implementation

> [!NOTE]  
> Put whatever is the current implementation of the solver here.
