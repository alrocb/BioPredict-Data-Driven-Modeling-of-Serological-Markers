"""
Output Capture Utility

This module provides a class to capture terminal output while also writing
to a log file. It is useful for capturing run-time logs of an analysis or model training.
"""

import sys
from io import StringIO

class OutputCapture:
    """
    Capture and log terminal output to a file.

    Attributes:
        terminal (stream): The original stdout stream.
        logfile (file object): File stream to write logs.
        capture_buffer (StringIO): Buffer to store captured output.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, 'w', encoding='utf-8')
        self.capture_buffer = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.capture_buffer.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()

    def get_captured_content(self):
        """Return the content captured so far."""
        return self.capture_buffer.getvalue()
