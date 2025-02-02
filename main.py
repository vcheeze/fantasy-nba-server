from asyncio import gather
from datetime import date
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from httpx_client import HTTPXClient
from ortools.linear_solver import pywraplp
from pydantic import BaseModel
from solver import TeamOptimizer
from typing import Dict, List, Optional, Tuple


httpx_client = HTTPXClient()
app = FastAPI()

origins = ["http://localhost:3000", "https://fantasy-nba.vercel.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/optimize")
async def optimize(gameweeks: List[int], points_column: Optional[str] = "form"):
    try:
        async_client = httpx_client()
        data_response = await async_client.get(
            "https://nbafantasy.nba.com/api/bootstrap-static/"
        )
        data = data_response.json()

        # Validate and collect all events first
        events = []
        for gameweek_id in gameweeks:
            event = next(
                (item for item in data["phases"] if item["id"] == gameweek_id), None
            )
            if not event:
                raise HTTPException(
                    status_code=404, detail=f"Event with ID {gameweek_id} not found"
                )
            events.append(event)

        # Fetch all fixtures in parallel
        fixture_tasks = [
            async_client.get(
                f"https://nbafantasy.nba.com/api/fixtures/?phase={gameweek_id}"
            )
            for gameweek_id in gameweeks
        ]
        fixture_responses = await gather(*fixture_tasks)

        # Combine all fixtures
        all_fixtures = []
        for response in fixture_responses:
            all_fixtures.extend(response.json())

        # Create list of all event numbers
        event_numbers = []
        for event in events:
            event_numbers.extend(range(event["start_event"], event["stop_event"] + 1))
        optimizer = TeamOptimizer(data, all_fixtures)
        result = optimizer.optimize(gamedays=event_numbers, points_column=points_column)
        return result
    except Exception as e:
        print(f"error :>> {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    httpx_client.start()


@app.on_event("shutdown")
async def shutdown_event():
    await httpx_client.stop()
