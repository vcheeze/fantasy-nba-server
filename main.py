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

origins = ["http://localhost:3000", "https://fantasy-nba.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/optimize")
async def optimize(gamedays: List[int], points_column: Optional[str] = "form"):
    try:
        async_client = httpx_client()
        data_response = await async_client.get(
            "https://nbafantasy.nba.com/api/bootstrap-static/"
        )
        if data_response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch metadata.")
        data = data_response.json()

        # Extract phase IDs based on gamedays
        phase_ids = set()
        for phase in data.get("phases", []):
            if any(start <= day <= stop for day in gamedays for start, stop in [(phase["start_event"], phase["stop_event"])]):
                phase_ids.add(phase["id"])
        if not phase_ids:
            raise HTTPException(status_code=404, detail="No matching phases found for the given gamedays.")

        # Fetch all fixtures in parallel
        fixture_tasks = [
            async_client.get(
                f"https://nbafantasy.nba.com/api/fixtures/?phase={gameweek_id}"
            )
            for gameweek_id in phase_ids
        ]
        fixture_responses = await gather(*fixture_tasks)

        # Combine all fixtures
        all_fixtures = []
        for response in fixture_responses:
            all_fixtures.extend(response.json())

        optimizer = TeamOptimizer(data, all_fixtures)
        result = optimizer.optimize(gamedays=gamedays, points_column=points_column)
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
