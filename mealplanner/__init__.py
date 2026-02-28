"""MealPlanner AI - Smart meal planning with macro tracking."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "Config":
        from .config import Config
        return Config
    elif name == "MacroTargets":
        from .config import MacroTargets
        return MacroTargets
    elif name == "MealPlanConfig":
        from .config import MealPlanConfig
        return MealPlanConfig
    elif name in ("Recipe", "Ingredient", "Macros", "MealPlan", "DayPlan", 
                  "MealSlot", "MealType", "FlavorProfile", "PantryItem",
                  "ShoppingList", "ShoppingListItem"):
        from . import models
        return getattr(models, name)
    elif name == "RecipeLibrary":
        from .recipe_parser import RecipeLibrary
        return RecipeLibrary
    elif name == "parse_recipe_file":
        from .recipe_parser import parse_recipe_file
        return parse_recipe_file
    elif name == "Pantry":
        from .pantry import Pantry
        return Pantry
    elif name == "MealPlanner":
        from .planner import MealPlanner
        return MealPlanner
    elif name == "MacroTracker":
        from .macro_tracker import MacroTracker
        return MacroTracker
    elif name == "ShoppingListGenerator":
        from .shopping import ShoppingListGenerator
        return ShoppingListGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Config",
    "MacroTargets", 
    "MealPlanConfig",
    "Recipe",
    "Ingredient",
    "Macros",
    "MealPlan",
    "DayPlan",
    "MealSlot",
    "MealType",
    "FlavorProfile",
    "PantryItem",
    "ShoppingList",
    "ShoppingListItem",
    "RecipeLibrary",
    "parse_recipe_file",
    "Pantry",
    "MealPlanner",
    "MacroTracker",
    "ShoppingListGenerator",
]
