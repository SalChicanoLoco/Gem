"""
ModelAgent - ML analytics agent for water quality predictions and data analysis.
"""

from typing import Any
from .gemma_agent import GemmaAgent


class ModelAgent:
    """
    Agent responsible for ML analytics tasks, specifically water quality analysis.
    Integrated with local Gemma LLM for deep reasoning and predictive analytics.
    """

    def __init__(self, model_path: str | None = None, gemma_agent: GemmaAgent | None = None):
        """
        Initialize the ModelAgent.

        Args:
            model_path: Optional path to a pre-trained model file.
            gemma_agent: Optional GemmaAgent instance for LLM reasoning.
        """
        self.model_path = model_path
        self.model = None
        self.gemma = gemma_agent or GemmaAgent()

    def load_model(self) -> bool:
        """
        Load a pre-trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self.model_path:
            self.model = {"status": "loaded", "path": self.model_path}
            return True
        return False

    def predict_water_quality(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict water quality based on input parameters and Gemma LLM analysis.

        Args:
            data: Dictionary containing water quality parameters
                  (e.g., pH, turbidity, temperature, dissolved_oxygen).

        Returns:
            Dictionary with prediction results, quality score, and Gemma reasoning.
        """
        # Extract parameters with defaults
        ph = data.get("ph", 7.0)
        turbidity = data.get("turbidity", 1.0)
        temperature = data.get("temperature", 20.0)
        dissolved_oxygen = data.get("dissolved_oxygen", 8.0)

        # Rule-based scoring engine
        score = self._calculate_quality_score(ph, turbidity, temperature, dissolved_oxygen)
        quality_label = self._get_quality_label(score)

        # Enhance with Gemma local LLM analysis
        gemma_insight = self.gemma.analyze_telemetry({
            "ph": ph,
            "turbidity": turbidity,
            "temperature": temperature,
            "dissolved_oxygen": dissolved_oxygen,
            "score": score,
            "label": quality_label,
        })

        return {
            "quality_score": round(score, 2),
            "quality_label": quality_label,
            "parameters": {
                "ph": ph,
                "turbidity": turbidity,
                "temperature": temperature,
                "dissolved_oxygen": dissolved_oxygen,
            },
            "recommendations": self._get_recommendations(score, ph, turbidity),
            "gemma_reasoning": gemma_insight,
        }

    def _calculate_quality_score(
        self, ph: float, turbidity: float, temperature: float, dissolved_oxygen: float
    ) -> float:
        """
        Calculate water quality score based on parameters.

        Returns:
            Quality score between 0 and 100.
        """
        score = 100.0

        # pH scoring (optimal 6.5-8.5)
        if ph < 6.5 or ph > 8.5:
            score -= abs(ph - 7.5) * 10

        # Turbidity scoring (lower is better, < 5 NTU is good)
        if turbidity > 5:
            score -= (turbidity - 5) * 5
        elif turbidity > 1:
            score -= (turbidity - 1) * 2

        # Temperature scoring (optimal 15-25°C for most uses)
        if temperature < 15 or temperature > 25:
            score -= abs(temperature - 20) * 0.5

        # Dissolved oxygen scoring (higher is better, > 6 mg/L is good)
        if dissolved_oxygen < 6:
            score -= (6 - dissolved_oxygen) * 5

        return max(0.0, min(100.0, score))

    @staticmethod
    def _get_quality_label(score: float) -> str:
        """Convert quality score to human-readable label."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Critical"

    @staticmethod
    def _get_recommendations(score: float, ph: float, turbidity: float) -> list[str]:
        """Generate recommendations based on water quality parameters."""
        recommendations = []

        if score < 60:
            recommendations.append("Consider additional water treatment")

        if ph < 6.5:
            recommendations.append("pH is too low - consider alkaline treatment")
        elif ph > 8.5:
            recommendations.append("pH is too high - consider acid treatment")

        if turbidity > 5:
            recommendations.append("High turbidity - filtration recommended")

        if not recommendations:
            recommendations.append("Water quality is satisfactory")

        return recommendations

    def analyze_trends(self, historical_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze trends in historical water quality data.

        Args:
            historical_data: List of historical water quality measurements.

        Returns:
            Dictionary with trend analysis results.
        """
        if not historical_data:
            return {"error": "No historical data provided"}

        scores = [self.predict_water_quality(d)["quality_score"] for d in historical_data]

        return {
            "average_score": round(sum(scores) / len(scores), 2),
            "min_score": min(scores),
            "max_score": max(scores),
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
            "data_points": len(scores),
        }
