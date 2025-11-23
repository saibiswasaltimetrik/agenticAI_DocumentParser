"""
Logging utilities including Responsible AI logging.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..schemas.document import AgentDecision, AgentType


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Setup application logging.

    Args:
        log_level: Override log level from settings

    Returns:
        Configured root logger
    """
    level = getattr(logging, (log_level or settings.log_level).upper())

    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "app.log"),
        ],
    )

    # Suppress verbose boto3 logging
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class ResponsibleAILogger:
    """
    Logger for tracking agent decisions and maintaining an audit trail
    for responsible AI compliance.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the Responsible AI logger.

        Args:
            log_file: Path to the log file (defaults to settings)
        """
        self.log_dir = Path(settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / (log_file or settings.responsible_ai_log_file)
        self.decisions: List[AgentDecision] = []
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._logger = get_logger("ResponsibleAI")

    def log_decision(
        self,
        agent_type: AgentType,
        agent_name: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        retry_count: int = 0,
        error_message: Optional[str] = None,
    ) -> AgentDecision:
        """
        Log an agent decision.

        Args:
            agent_type: Type of agent making the decision
            agent_name: Name/identifier of the agent
            input_summary: Summary of input data
            output_summary: Summary of output/decision
            duration_ms: Processing duration in milliseconds
            confidence: Confidence score (0-1)
            reasoning: Explanation of the decision
            model_used: LLM model used
            tokens_used: Number of tokens consumed
            retry_count: Number of retries needed
            error_message: Error message if failed

        Returns:
            The logged AgentDecision
        """
        decision = AgentDecision(
            agent_type=agent_type,
            agent_name=agent_name,
            input_summary=input_summary[:500],  # Truncate long inputs
            output_summary=output_summary[:500],  # Truncate long outputs
            duration_ms=duration_ms,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
            tokens_used=tokens_used,
            retry_count=retry_count,
            error_message=error_message,
        )

        self.decisions.append(decision)
        self._logger.info(
            f"Agent Decision: {agent_type.value}/{agent_name} - "
            f"Duration: {duration_ms:.2f}ms, Confidence: {confidence}"
        )

        return decision

    def log_pii_metrics(
        self,
        total_detected: int,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> Dict[str, float]:
        """
        Log PII detection metrics for responsible AI tracking.

        Args:
            total_detected: Total PII entities detected
            true_positives: Correctly identified PII
            false_positives: Incorrectly flagged as PII
            false_negatives: Missed PII

        Returns:
            Calculated metrics
        """
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "total_detected": total_detected,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        self._logger.info(
            f"PII Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        return metrics

    def save(self) -> str:
        """
        Save all logged decisions to file.

        Returns:
            Path to the saved log file
        """
        log_data = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_decisions": len(self.decisions),
            "decisions": [d.model_dump() for d in self.decisions],
        }

        # Convert datetime objects for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=json_serializer)

        self._logger.info(f"Saved {len(self.decisions)} decisions to {self.log_file}")
        return str(self.log_file)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all logged decisions.

        Returns:
            Summary statistics
        """
        if not self.decisions:
            return {"total_decisions": 0}

        agent_counts = {}
        total_duration = 0.0
        error_count = 0

        for decision in self.decisions:
            agent_type = decision.agent_type.value
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
            total_duration += decision.duration_ms
            if decision.error_message:
                error_count += 1

        return {
            "session_id": self.session_id,
            "total_decisions": len(self.decisions),
            "decisions_by_agent": agent_counts,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(self.decisions),
            "error_count": error_count,
            "error_rate": error_count / len(self.decisions),
        }

    def clear(self):
        """Clear all logged decisions."""
        self.decisions = []
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
