"""Data models for MealPlanner AI."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Optional
from pathlib import Path
import re


class MealType(Enum):
    """Types of meals."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    DESSERT = "dessert"


class FlavorProfile(Enum):
    """Flavor profiles for mood-based filtering."""
    SPICY = "spicy"
    SAVORY = "savory"
    SWEET = "sweet"
    SOUR = "sour"
    UMAMI = "umami"
    FRESH = "fresh"
    RICH = "rich"
    LIGHT = "light"


@dataclass
class Macros:
    """Nutritional macro information."""
    calories: float = 0.0
    protein: float = 0.0  # grams
    carbs: float = 0.0    # grams
    fat: float = 0.0      # grams
    
    def __add__(self, other: "Macros") -> "Macros":
        return Macros(
            calories=self.calories + other.calories,
            protein=self.protein + other.protein,
            carbs=self.carbs + other.carbs,
            fat=self.fat + other.fat,
        )
    
    def __mul__(self, factor: float) -> "Macros":
        return Macros(
            calories=self.calories * factor,
            protein=self.protein * factor,
            carbs=self.carbs * factor,
            fat=self.fat * factor,
        )
    
    def to_dict(self) -> dict:
        return {
            "calories": round(self.calories, 1),
            "protein": round(self.protein, 1),
            "carbs": round(self.carbs, 1),
            "fat": round(self.fat, 1),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Macros":
        return cls(
            calories=float(data.get("calories", 0) or 0),
            protein=float(data.get("protein", 0) or 0),
            carbs=float(data.get("carbs", 0) or 0),
            fat=float(data.get("fat", 0) or 0),
        )


@dataclass
class Ingredient:
    """A recipe ingredient with optional quantity and unit."""
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    notes: Optional[str] = None
    raw_text: str = ""
    
    def __str__(self) -> str:
        if self.quantity and self.unit:
            return f"{self.quantity} {self.unit} {self.name}"
        elif self.quantity:
            return f"{self.quantity} {self.name}"
        return self.name
    
    @classmethod
    def parse(cls, text: str) -> "Ingredient":
        """Parse an ingredient string into structured data."""
        raw_text = text.strip()
        
        # Remove checkbox markdown
        text = re.sub(r"^\s*-\s*\[\s*\]\s*", "", text).strip()
        
        # Common patterns:
        # "93/7 ground beef, 450 g"
        # "2 tablespoons extra-virgin olive oil"
        # "Salt, to taste"
        # "Eggs, 4-12"
        
        quantity = None
        unit = None
        name = text
        notes = None
        
        # Try to extract quantity and unit
        # Pattern: "number unit name" or "name, number unit"
        
        # Pattern 1: "name, quantity unit" (e.g., "ground beef, 450 g")
        match = re.match(r"^(.+?),\s*([\d./\-]+)\s*([a-zA-Z]+)?\s*$", text)
        if match:
            name = match.group(1).strip()
            qty_str = match.group(2)
            unit = match.group(3)
            # Handle ranges like "4-12" by taking the average
            if "-" in qty_str:
                parts = qty_str.split("-")
                try:
                    quantity = (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    pass
            else:
                try:
                    # Handle fractions like "1/2"
                    if "/" in qty_str:
                        parts = qty_str.split("/")
                        quantity = float(parts[0]) / float(parts[1])
                    else:
                        quantity = float(qty_str)
                except ValueError:
                    pass
        
        # Pattern 2: "quantity unit name" (e.g., "2 tablespoons olive oil")
        if not match:
            match = re.match(r"^([\d./]+)\s+(\w+)\s+(.+)$", text)
            if match:
                qty_str = match.group(1)
                unit = match.group(2)
                name = match.group(3).strip()
                try:
                    if "/" in qty_str:
                        parts = qty_str.split("/")
                        quantity = float(parts[0]) / float(parts[1])
                    else:
                        quantity = float(qty_str)
                except ValueError:
                    pass
        
        # Pattern 3: "quantity name" (e.g., "4 eggs")
        if not match:
            match = re.match(r"^([\d./]+)\s+(.+)$", text)
            if match:
                qty_str = match.group(1)
                name = match.group(2).strip()
                try:
                    if "/" in qty_str:
                        parts = qty_str.split("/")
                        quantity = float(parts[0]) / float(parts[1])
                    else:
                        quantity = float(qty_str)
                except ValueError:
                    pass
        
        # Extract notes in parentheses
        notes_match = re.search(r"\(([^)]+)\)", name)
        if notes_match:
            notes = notes_match.group(1)
            name = re.sub(r"\s*\([^)]+\)\s*", " ", name).strip()
        
        # Clean up common suffixes
        name = re.sub(r",\s*(to taste|as needed|optional)\s*$", "", name, flags=re.IGNORECASE)
        
        return cls(
            name=name.strip(" ,"),
            quantity=quantity,
            unit=unit,
            notes=notes,
            raw_text=raw_text,
        )


@dataclass
class Recipe:
    """A parsed recipe with all metadata."""
    name: str
    source_path: Path
    source_url: Optional[str] = None
    
    # Timing
    prep_time: Optional[timedelta] = None
    cook_time: Optional[timedelta] = None
    total_time: Optional[timedelta] = None
    
    # Yield and servings
    yields: Optional[str] = None
    servings: int = 1
    
    # Nutrition (per serving)
    macros: Macros = field(default_factory=Macros)
    
    # Content
    ingredients: list[Ingredient] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    flavor_profiles: list[FlavorProfile] = field(default_factory=list)
    meal_types: list[MealType] = field(default_factory=list)
    is_ai_suggested: bool = False  # True for snacks/meals not from recipe library
    
    # For meal prep optimization
    can_meal_prep: bool = False
    prep_ahead_days: int = 0
    
    def get_total_macros(self, servings: int = 1) -> Macros:
        """Get macros for a specific number of servings."""
        return self.macros * servings
    
    def get_ingredient_names(self) -> list[str]:
        """Get list of ingredient names (normalized)."""
        return [ing.name.lower() for ing in self.ingredients]
    
    def matches_mood(self, moods: list[str]) -> bool:
        """Check if recipe matches any of the specified moods."""
        if not moods:
            return True
        recipe_profiles = [p.value for p in self.flavor_profiles]
        return any(mood.lower() in recipe_profiles for mood in moods)
    
    def to_summary(self) -> str:
        """Generate a brief summary of the recipe."""
        time_str = ""
        if self.total_time:
            minutes = int(self.total_time.total_seconds() / 60)
            time_str = f" ({minutes} min)"
        
        return f"{self.name}{time_str} - {self.macros.calories:.0f} cal, {self.macros.protein:.1f}g protein"


@dataclass
class PantryItem:
    """An item in the pantry inventory."""
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    category: Optional[str] = None  # e.g., "protein", "produce", "pantry staple"
    expiration_date: Optional[str] = None
    
    @property
    def is_staple(self) -> bool:
        """Items without quantities are assumed always in stock (staples)."""
        return self.quantity is None
    
    def matches(self, ingredient_name: str) -> bool:
        """Check if this pantry item matches an ingredient name."""
        return (
            self.name.lower() in ingredient_name.lower() or
            ingredient_name.lower() in self.name.lower()
        )


@dataclass
class MealSlot:
    """A single meal slot in a meal plan."""
    day: int  # 1-7 for a week
    meal_type: MealType
    recipe: Optional[Recipe] = None
    servings: float = 1.0
    notes: Optional[str] = None
    
    def get_macros(self) -> Macros:
        """Get macros for this meal slot."""
        if self.recipe:
            return self.recipe.macros * self.servings
        return Macros()


@dataclass
class DayPlan:
    """A day's worth of meals."""
    day: int
    meals: list[MealSlot] = field(default_factory=list)
    
    def get_total_macros(self) -> Macros:
        """Get total macros for the day."""
        total = Macros()
        for meal in self.meals:
            total = total + meal.get_macros()
        return total
    
    def get_recipes(self) -> list[Recipe]:
        """Get all recipes for the day."""
        return [m.recipe for m in self.meals if m.recipe]


@dataclass
class MealPlan:
    """A complete meal plan (typically a week)."""
    days: list[DayPlan] = field(default_factory=list)
    
    def get_all_recipes(self) -> list[Recipe]:
        """Get all unique recipes in the plan."""
        seen = set()
        recipes = []
        for day in self.days:
            for recipe in day.get_recipes():
                if recipe.name not in seen:
                    seen.add(recipe.name)
                    recipes.append(recipe)
        return recipes
    
    def get_all_ingredients(self) -> list[tuple[Ingredient, float]]:
        """Get all ingredients with quantities (scaled by servings)."""
        ingredients = []
        for day in self.days:
            for meal in day.meals:
                if meal.recipe:
                    for ing in meal.recipe.ingredients:
                        scaled_qty = ing.quantity * meal.servings if ing.quantity else None
                        scaled_ing = Ingredient(
                            name=ing.name,
                            quantity=scaled_qty,
                            unit=ing.unit,
                            notes=ing.notes,
                            raw_text=ing.raw_text,
                        )
                        ingredients.append((scaled_ing, meal.servings))
        return ingredients
    
    def get_average_daily_macros(self) -> Macros:
        """Get average daily macros across the plan."""
        if not self.days:
            return Macros()
        total = Macros()
        for day in self.days:
            total = total + day.get_total_macros()
        return total * (1 / len(self.days))


@dataclass
class ShoppingListItem:
    """An item on the shopping list."""
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    category: str = "other"
    in_pantry: bool = False
    pantry_quantity: Optional[float] = None
    needed_quantity: Optional[float] = None
    
    def __str__(self) -> str:
        if self.quantity and self.unit:
            return f"{self.name}: {self.quantity:.1f} {self.unit}"
        elif self.quantity:
            return f"{self.name}: {self.quantity:.1f}"
        return self.name


@dataclass
class ShoppingList:
    """A shopping list generated from a meal plan."""
    items: list[ShoppingListItem] = field(default_factory=list)
    
    def get_by_category(self) -> dict[str, list[ShoppingListItem]]:
        """Group items by category."""
        categories: dict[str, list[ShoppingListItem]] = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        return categories
    
    def to_markdown(self) -> str:
        """Convert shopping list to markdown format."""
        lines = ["# Shopping List\n"]
        
        by_category = self.get_by_category()
        for category, items in sorted(by_category.items()):
            lines.append(f"\n## {category.title()}\n")
            for item in sorted(items, key=lambda x: x.name):
                if not item.in_pantry:
                    lines.append(f"- [ ] {item}")
        
        return "\n".join(lines)
