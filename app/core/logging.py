import logging

def configure_logging():
    """
    Simple logging setup for the RAG application.
    Logs are printed to the console with timestamps and log levels.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info("✅ Logging initialized successfully.")
