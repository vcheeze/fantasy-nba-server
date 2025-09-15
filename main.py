from asyncio import gather
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from httpx_client import HTTPXClient
from solver import TeamOptimizer, Transfers
from typing import Dict, List, Optional, Set


httpx_client = HTTPXClient()
app = FastAPI(
    title="NBA Fantasy Optimizer API",
    description="API for optimizing NBA fantasy teams",
    version="1.0.0",
)

origins = ["http://localhost:3000", "https://fantasy-nba.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_phase_ids_for_gamedays(data: Dict, gamedays: List[int]) -> Set[int]:
    """
    Extract phase IDs that contain the specified gamedays.

    Args:
        data: The bootstrap data containing phases
        gamedays: List of gameday IDs to find phases for

    Returns:
        Set of phase IDs that contain the specified gamedays
    """
    phase_ids = set()
    for phase in data.get("phases", []):
        start, stop = phase["start_event"], phase["stop_event"]
        if any(start <= day <= stop for day in gamedays):
            phase_ids.add(phase["id"])
    return phase_ids


async def fetch_fixtures(async_client, phase_ids: Set[int]) -> List[Dict]:
    """
    Fetch fixtures for all specified phase IDs in parallel.

    Args:
        async_client: HTTP client for making requests
        phase_ids: Set of phase IDs to fetch fixtures for

    Returns:
        List of all fixtures from the specified phases
    """
    fixture_tasks = [
        async_client.get(f"https://nbafantasy.nba.com/api/fixtures/?phase={phase_id}")
        for phase_id in phase_ids
    ]
    fixture_responses = await gather(*fixture_tasks)

    all_fixtures = []
    for response in fixture_responses:
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch fixtures for phase",
            )
        all_fixtures.extend(response.json())

    return all_fixtures


@app.post("/optimize")
async def optimize(
    gamedays: List[int],
    points_column: Optional[str] = "form",
    picks: Optional[List[Dict]] = None,
    transfers: Optional[Transfers] = None,
):
    """
    Optimize a fantasy team based on specified gamedays and constraints.

    Args:
        gamedays: List of gameday IDs to optimize for
        points_column: Metric to use for scoring (default: "form")
        picks: Current squad selection (optional)
        transfers: Transfer constraints (optional)

    Returns:
        Optimized team selection
    """
    try:
        async_client = httpx_client()

        # Fetch bootstrap data
        data_response = await async_client.get(
            "https://nbafantasy.nba.com/api/bootstrap-static/"
        )
        if data_response.status_code != 200:
            raise HTTPException(
                status_code=data_response.status_code,
                detail="Failed to fetch metadata.",
            )
        data = data_response.json()

        # Get phase IDs for the specified gamedays
        phase_ids = get_phase_ids_for_gamedays(data, gamedays)
        if not phase_ids:
            raise HTTPException(
                status_code=404,
                detail="No matching phases found for the given gamedays.",
            )

        # Fetch fixtures for all relevant phases
        all_fixtures = await fetch_fixtures(async_client, phase_ids)

        # Run optimization
        optimizer = TeamOptimizer(
            players=data["elements"],
            fixtures=all_fixtures,
            phases=data["phases"],
            teams=data["teams"],
            scoring_metric=points_column,
            current_squad=picks,
            transfers=transfers,
        )
        print(f'picks:>> {picks}')
        result = optimizer.optimize(
            event_ids=gamedays,
            current_squad=picks,
        )
        return result

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        print(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "NBA Fantasy Optimizer"}


@app.on_event("startup")
async def startup_event():
    """Initialize the HTTP client on application startup."""
    httpx_client.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await httpx_client.stop()
