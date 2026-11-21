"""Dashboard wrapper for simulated event stream generation."""
from src.streaming.event_simulator import (
    DEFAULT_EVENT_DIR,
    generate_simulated_stream_events,
    write_events_to_directory,
    run_event_generator,
)

__all__ = [
    "DEFAULT_EVENT_DIR",
    "generate_simulated_stream_events",
    "write_events_to_directory",
    "run_event_generator",
]
