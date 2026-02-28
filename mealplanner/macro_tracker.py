"""Macro tracking and nutritional analysis."""

from dataclasses import dataclass
from typing import Optional

from .models import Macros, Recipe, MealPlan, DayPlan
from .config import MacroTargets


@dataclass
class MacroAnalysis:
    """Analysis of macro targets vs actual."""
    target: Macros
    actual: Macros
    
    @property
    def calories_diff(self) -> float:
        return self.actual.calories - self.target.calories
    
    @property
    def protein_diff(self) -> float:
        return self.actual.protein - self.target.protein
    
    @property
    def carbs_diff(self) -> float:
        return self.actual.carbs - self.target.carbs
    
    @property
    def fat_diff(self) -> float:
        return self.actual.fat - self.target.fat
    
    @property
    def calories_pct(self) -> float:
        return (self.actual.calories / self.target.calories * 100) if self.target.calories else 0
    
    @property
    def protein_pct(self) -> float:
        return (self.actual.protein / self.target.protein * 100) if self.target.protein else 0
    
    @property
    def carbs_pct(self) -> float:
        return (self.actual.carbs / self.target.carbs * 100) if self.target.carbs else 0
    
    @property
    def fat_pct(self) -> float:
        return (self.actual.fat / self.target.fat * 100) if self.target.fat else 0
    
    def is_within_target(self, tolerance_pct: float = 10) -> bool:
        """Check if all macros are within tolerance of target."""
        return all([
            abs(self.calories_pct - 100) <= tolerance_pct,
            abs(self.protein_pct - 100) <= tolerance_pct,
            abs(self.carbs_pct - 100) <= tolerance_pct,
            abs(self.fat_pct - 100) <= tolerance_pct,
        ])
    
    def get_deficiencies(self) -> list[str]:
        """Get list of macro deficiencies."""
        deficiencies = []
        if self.protein_pct < 90:
            deficiencies.append(f"Protein: {self.protein_diff:.1f}g short")
        if self.calories_pct < 90:
            deficiencies.append(f"Calories: {abs(self.calories_diff):.0f} short")
        return deficiencies
    
    def get_excesses(self) -> list[str]:
        """Get list of macro excesses."""
        excesses = []
        if self.calories_pct > 110:
            excesses.append(f"Calories: {self.calories_diff:.0f} over")
        if self.fat_pct > 120:
            excesses.append(f"Fat: {self.fat_diff:.1f}g over")
        if self.carbs_pct > 120:
            excesses.append(f"Carbs: {self.carbs_diff:.1f}g over")
        return excesses
    
    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "📊 Macro Analysis",
            f"  Calories: {self.actual.calories:.0f} / {self.target.calories:.0f} ({self.calories_pct:.0f}%)",
            f"  Protein:  {self.actual.protein:.1f}g / {self.target.protein:.1f}g ({self.protein_pct:.0f}%)",
            f"  Carbs:    {self.actual.carbs:.1f}g / {self.target.carbs:.1f}g ({self.carbs_pct:.0f}%)",
            f"  Fat:      {self.actual.fat:.1f}g / {self.target.fat:.1f}g ({self.fat_pct:.0f}%)",
        ]
        
        deficiencies = self.get_deficiencies()
        if deficiencies:
            lines.append("\n⚠️  Deficiencies:")
            for d in deficiencies:
                lines.append(f"    - {d}")
        
        excesses = self.get_excesses()
        if excesses:
            lines.append("\n⚠️  Excesses:")
            for e in excesses:
                lines.append(f"    - {e}")
        
        if self.is_within_target():
            lines.append("\n✅ All macros within target!")
        
        return "\n".join(lines)


class MacroTracker:
    """Track and analyze macros against targets."""
    
    def __init__(self, targets: MacroTargets):
        self.targets = targets
        self.target_macros = Macros(
            calories=targets.calories,
            protein=targets.protein,
            carbs=targets.carbs,
            fat=targets.fat,
        )
    
    def analyze_recipe(self, recipe: Recipe, servings: float = 1) -> MacroAnalysis:
        """Analyze a single recipe against daily targets."""
        actual = recipe.macros * servings
        return MacroAnalysis(target=self.target_macros, actual=actual)
    
    def analyze_day(self, day_plan: DayPlan) -> MacroAnalysis:
        """Analyze a day's meals against daily targets."""
        actual = day_plan.get_total_macros()
        return MacroAnalysis(target=self.target_macros, actual=actual)
    
    def analyze_meal_plan(self, meal_plan: MealPlan) -> list[MacroAnalysis]:
        """Analyze each day in a meal plan."""
        return [self.analyze_day(day) for day in meal_plan.days]
    
    def get_average_analysis(self, meal_plan: MealPlan) -> MacroAnalysis:
        """Get average daily macro analysis for a meal plan."""
        avg_macros = meal_plan.get_average_daily_macros()
        return MacroAnalysis(target=self.target_macros, actual=avg_macros)
    
    def suggest_adjustments(self, analysis: MacroAnalysis) -> list[str]:
        """Suggest adjustments to hit macro targets."""
        suggestions = []
        
        # Protein suggestions
        if analysis.protein_pct < 90:
            deficit = self.targets.protein - analysis.actual.protein
            suggestions.append(
                f"Add ~{deficit:.0f}g protein: Greek yogurt, chicken, or protein shake"
            )
        
        # Calorie suggestions
        if analysis.calories_pct < 85:
            deficit = self.targets.calories - analysis.actual.calories
            suggestions.append(
                f"Add ~{deficit:.0f} calories: nuts, avocado, or healthy snacks"
            )
        elif analysis.calories_pct > 115:
            excess = analysis.actual.calories - self.targets.calories
            suggestions.append(
                f"Reduce ~{excess:.0f} calories: smaller portions or swap ingredients"
            )
        
        # Fat suggestions
        if analysis.fat_pct > 130:
            suggestions.append(
                "Consider lower-fat cooking methods: grilling, baking, or steaming"
            )
        
        # Carb suggestions
        if analysis.carbs_pct > 130:
            suggestions.append(
                "Consider reducing carbs: smaller grain portions or veggie swaps"
            )
        
        return suggestions
    
    def score_recipe_fit(self, recipe: Recipe, current_day_macros: Macros) -> float:
        """
        Score how well a recipe fits the remaining macro budget for the day.
        Returns 0-100 score.
        """
        remaining = Macros(
            calories=max(0, self.targets.calories - current_day_macros.calories),
            protein=max(0, self.targets.protein - current_day_macros.protein),
            carbs=max(0, self.targets.carbs - current_day_macros.carbs),
            fat=max(0, self.targets.fat - current_day_macros.fat),
        )
        
        recipe_macros = recipe.macros
        
        # Calculate how well each macro fits
        scores = []
        
        # Protein: reward meeting target, penalize going over slightly
        if remaining.protein > 0:
            protein_score = min(100, (recipe_macros.protein / remaining.protein) * 100)
            if recipe_macros.protein > remaining.protein * 1.2:
                protein_score *= 0.8  # Slight penalty for going over
        else:
            protein_score = 50 if recipe_macros.protein < 20 else 30
        scores.append(protein_score * 1.5)  # Weight protein higher
        
        # Calories: reward staying within budget
        if remaining.calories > 0:
            cal_ratio = recipe_macros.calories / remaining.calories
            if cal_ratio <= 1:
                cal_score = 100 - (1 - cal_ratio) * 30  # Reward being close
            else:
                cal_score = max(0, 100 - (cal_ratio - 1) * 50)  # Penalize going over
        else:
            cal_score = 30 if recipe_macros.calories < 300 else 10
        scores.append(cal_score)
        
        # Fat and carbs: moderate penalties for going over
        if remaining.fat > 0:
            fat_ratio = recipe_macros.fat / remaining.fat
            fat_score = max(0, 100 - max(0, fat_ratio - 1) * 40)
        else:
            fat_score = 50
        scores.append(fat_score * 0.7)
        
        if remaining.carbs > 0:
            carb_ratio = recipe_macros.carbs / remaining.carbs
            carb_score = max(0, 100 - max(0, carb_ratio - 1) * 40)
        else:
            carb_score = 50
        scores.append(carb_score * 0.7)
        
        # Weighted average
        total_weight = 1.5 + 1 + 0.7 + 0.7
        return sum(scores) / total_weight
    
    def find_complementary_recipes(
        self,
        recipes: list[Recipe],
        current_macros: Macros,
        top_n: int = 5
    ) -> list[tuple[Recipe, float]]:
        """Find recipes that best complement current day's macros."""
        scored = [
            (recipe, self.score_recipe_fit(recipe, current_macros))
            for recipe in recipes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
