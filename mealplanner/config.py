"""Configuration management for MealPlanner AI."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MacroTargets:
    """Daily macro nutritional targets."""
    calories: float = 2000.0
    protein: float = 150.0  # grams
    carbs: float = 200.0    # grams
    fat: float = 65.0       # grams
    
    def __post_init__(self):
        """Validate targets are positive."""
        for field_name in ['calories', 'protein', 'carbs', 'fat']:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass 
class MealPlanConfig:
    """Configuration for meal planning preferences."""
    days: int = 7
    meals_per_day: int = 3
    max_prep_time_minutes: int = 60
    max_cook_time_minutes: int = 90
    prefer_meal_prep: bool = True
    variety_weight: float = 0.7  # 0-1, how much to prioritize variety
    pantry_weight: float = 0.9   # 0-1, how much to prioritize using pantry items (higher = prefer pantry recipes)
    
    # Preferences / mood
    mood_preferences: list[str] = field(default_factory=list)  # e.g., ["spicy", "savory"]
    cuisine_preferences: list[str] = field(default_factory=list)  # e.g., ["mexican", "asian"]
    excluded_ingredients: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Main application configuration."""
    gemini_api_key: str
    recipes_path: Path
    pantry_path: Path
    
    # Optional paths
    calendar_path: Optional[Path] = None
    output_path: Optional[Path] = None
    
    # Nested configs
    macro_targets: MacroTargets = field(default_factory=MacroTargets)
    meal_plan_config: MealPlanConfig = field(default_factory=MealPlanConfig)
    
    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        # Load .env file if it exists - check multiple locations
        if dotenv_path is None:
            # Try project root first, then package directory
            project_root = Path(__file__).parent.parent / ".env"
            package_dir = Path(__file__).parent / ".env"
            cwd = Path.cwd() / ".env"
            
            for path in [cwd, project_root, package_dir]:
                if path.exists():
                    dotenv_path = path
                    break
        
        if dotenv_path and dotenv_path.exists():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        recipes_path = Path(os.environ.get(
            "RECIPES_PATH", 
            Path.home() / "Documents/obsidian_vault/recipes"
        ))
        pantry_path = Path(os.environ.get(
            "PANTRY_PATH",
            Path.home() / "Documents/obsidian_vault/pantry.md"
        ))
        
        calendar_path = os.environ.get("CALENDAR_PATH")
        output_path = os.environ.get("OUTPUT_PATH")
        
        return cls(
            gemini_api_key=gemini_api_key,
            recipes_path=recipes_path,
            pantry_path=pantry_path,
            calendar_path=Path(calendar_path) if calendar_path else None,
            output_path=Path(output_path) if output_path else None,
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is not set")
        
        if not self.recipes_path.exists():
            errors.append(f"Recipes path does not exist: {self.recipes_path}")
        
        # Pantry path doesn't need to exist - we can create it
        
        return errors
