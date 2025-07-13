import logging


class LogColors:
    """
    ANSI escape codes for coloring log messages in the terminal.
    """

    DEBUG = "\033[90m"  # Grey
    INFO = "\033[94m"  # Blue
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    CRITICAL = "\033[95m"  # Magenta
    RESET = "\033[0m"  # Reset to default terminal color


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color to log messages
    based on their severity level (INFO, WARNING, ERROR, etc.).
    """

    def format(self, record):
        """
        Format the log message with the appropriate color.

        Args:
            record (LogRecord): The log record to be formatted.

        Returns:
            str: The formatted, colorized log message.
        """
        color = ""
        # Assign a color based on the log level
        if record.levelno == logging.DEBUG:
            color = LogColors.DEBUG
        elif record.levelno == logging.INFO:
            color = LogColors.INFO
        elif record.levelno == logging.WARNING:
            color = LogColors.WARNING
        elif record.levelno == logging.ERROR:
            color = LogColors.ERROR
        elif record.levelno == logging.CRITICAL:
            color = LogColors.CRITICAL

        # Wrap the message in color codes
        record.msg = f"{color}{record.getMessage()}{LogColors.RESET}"

        # Call the parent class formatter
        return super().format(record)


def get_logger(name: str):
    """
    Create or retrieve a logger with a colorized console output.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        # Create a stream handler to output logs to the output stream—by default, this is sys.stderr (your terminal).
        handler = logging.StreamHandler()

        # Attach the custom color formatter
        handler.setFormatter(ColoredFormatter("%(asctime)s - %(message)s"))
        # Add the handler to the logger
        logger.addHandler(handler)
    return logger
