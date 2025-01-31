import httpx
import logging

logging.basicConfig(
    level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


class HTTPXClient:
    async_client = None

    def start(self):
        """Instantiate the client. Call from the FastAPI startup hook."""
        self.async_client = httpx.AsyncClient()
        LOGGER.info(f"httpx AsyncClient instantiated. Id {id(self.async_client)}")

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        LOGGER.info(
            f"httpx async_client.is_closed(): {self.async_client.is_closed} - Now close it. Id (will be unchanged): {id(self.async_client)}"
        )
        await self.async_client.aclose()
        LOGGER.info(
            f"httpx async_client.is_closed(): {self.async_client.is_closed}. Id (will be unchanged): {id(self.async_client)}"
        )
        self.async_client = None
        LOGGER.info("httpx AsyncClient closed")

    def __call__(self):
        """Calling the instantiated HTTPXClientWrapper returns the wrapped singleton."""
        # Ensure we don't use it if not started / running
        assert self.async_client is not None
        LOGGER.info(
            f"httpx async_client.is_closed(): {self.async_client.is_closed}. Id (will be unchanged): {id(self.async_client)}"
        )
        return self.async_client
