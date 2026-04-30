"""
Context manager to track execution time and statistics (like LLM usage) for pipeline stages.
Updates a metadata dictionary with timing information and formatted readable durations.
"""

import time
import datetime
from typing import Dict, Any, Optional

class ExecutionMeasures:
    """
    Context manager to track execution time and stats (like LLM usage) of pipeline stages.
    Updates a 'execution_measures' dictionary within the provided metadata.
    """
    def __init__(
        self, 
        measures_dict: Dict[str, Any], 
        stage_name: str, 
        module_name: str = "unknown", 
        initial_duration: float = 0.0
    ):
        self.measures_dict = measures_dict
        self.stage_name = stage_name
        self.module_name = module_name
        self.initial_duration = initial_duration
        self.start_time = None
        self.extra_stats = {}

    def __enter__(self) -> 'ExecutionMeasures':
        self.start_time = time.time()
        
        # Initialize entry with running status
        self.measures_dict[self.stage_name] = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module": self.module_name,
            "status": "running",
            "duration_seconds": self.initial_duration
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_duration = self.get_current_duration()
        status = "completed" if exc_type is None else "failed"
        
        # Update final stats
        update_payload = {
            "duration_seconds": round(total_duration, 2),
            "status": status
        }
        # Merge any extra stats collected (like final LLM stats)
        update_payload.update(self.extra_stats)
        
        self.measures_dict[self.stage_name].update(update_payload)

    def get_current_duration(self) -> float:
        """Returns the total duration (initial + elapsed) up to this moment."""
        if self.start_time is None:
            return self.initial_duration
        return self.initial_duration + (time.time() - self.start_time)

    def add_llm_stats(self, stats: Dict[str, Any]) -> None:
        """Registers LLM statistics to be saved along with execution time."""
        self.extra_stats["llm_stats"] = stats

    @staticmethod
    def format_seconds(seconds: float) -> str:
        """Formats seconds into HH:MM:SS string."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _add_readable_times(measures: Dict[str, Any]) -> None:
        """Adds readable duration strings to execution measures."""
        for key, value in list(measures.items()):
            if isinstance(value, dict):
                if "duration_seconds" in value:
                    value["duration_readable"] = ExecutionMeasures.format_seconds(value["duration_seconds"])
            elif key == "total_pipeline_duration" and isinstance(value, (int, float)):
                measures["total_pipeline_duration_readable"] = ExecutionMeasures.format_seconds(value)

    @staticmethod
    def calculate_final_times(measures_dict: Dict[str, Any]) -> None:
        """
        Calculates the total execution time across all recorded stages.
        Adds a 'total_pipeline_duration' entry.
        Also adds human-readable duration strings to all time entries.
        """
        total_time = 0.0
        for stage, data in measures_dict.items():
            # Sum duration of all stages that have a duration_seconds field
            if isinstance(data, dict) and "duration_seconds" in data:
                total_time += data["duration_seconds"]
        
        measures_dict["total_pipeline_duration"] = round(total_time, 2)
        
        # Add readable times
        ExecutionMeasures._add_readable_times(measures_dict)

        # Make key X go to the end of the dictionary
        if "total_pipeline_duration" in measures_dict:
            measures_dict["total_pipeline_duration"] = measures_dict.pop("total_pipeline_duration")
        if "total_pipeline_duration_readable" in measures_dict:
            measures_dict["total_pipeline_duration_readable"] = measures_dict.pop("total_pipeline_duration_readable")


        