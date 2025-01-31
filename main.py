from datetime import date
from enum import Enum
from fastapi import FastAPI, HTTPException
from httpx_client import HTTPXClient
from ortools.linear_solver import pywraplp
from pydantic import BaseModel
from solver import TeamOptimizer
from typing import Dict, List, Optional, Tuple

httpx_client = HTTPXClient()
app = FastAPI()


@app.post("/optimize")
async def optimize(gameweeks: List[int], points_column: Optional[str] = "form"):
    try:
        async_client = httpx_client()
        data_response = await async_client.get("https://nbafantasy.nba.com/api/bootstrap-static/")
        data = data_response.json()
        fixtures_response = await async_client.get(f'https://nbafantasy.nba.com/api/fixtures/?phase={gameweeks[0]}')
        fixtures = fixtures_response.json()
        optimizer = TeamOptimizer(data, fixtures)
        result = optimizer.optimize(gamedays=[93, 94, 95, 96, 97, 98, 99], points_column=points_column)
        return result
    except Exception as e:
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