import os
import smtplib
import ssl
from contextlib import asynccontextmanager

import redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from tortoise import Tortoise

from src.agents.chatbot.agent import chatbot_graph
from src.config import TORTOISE_CONFIG
from src.models.app_state import app_state
from src.models.services.mailer import Mailer


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_state.db_conn = await AsyncConnection.connect(conninfo=os.getenv("DATABASE_URL"), autocommit=True)
        app_state.memory = AsyncPostgresSaver(conn=app_state.db_conn)

        graph, tracer = await chatbot_graph()
        app_state.graph = graph
        app_state.tracer = tracer

        await app_state.memory.setup()
        await Tortoise.init(
            config=TORTOISE_CONFIG
        )
        await Tortoise.generate_schemas(safe=True)

        app_state.redis_store = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            decode_responses=True
        )

        app_state.redis_store.ping()

        # app_state.mailer = Mailer()

        print("Application initialized")
        yield
    except RedisConnectionError as e:
        print(f"Could not connect to Redis server! {e}")
        raise e
    except RedisError as e:
        print(f"Error in Redis Database occurred! {e}")
        raise e
    except Exception as e:
        print(f"Application initialization failed! {e}")
        raise e
    finally:
        await Tortoise.close_connections()
        await app_state.db_conn.close()

        if app_state.redis_store is not None:
            app_state.redis_store.close()

        if app_state.mailer is not None:
            app_state.mailer.disconnect()