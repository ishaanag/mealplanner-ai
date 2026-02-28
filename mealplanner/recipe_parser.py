"""Recipe parser for markdown files with YAML frontmatter."""

import re
from pathlib import Path
from datetime import timedelta
from typing import Optional
import frontmatter

from .models import (
    Recipe, Ingredient, Macros, MealType, FlavorProfile
)


def parse_iso_duration(duration_str: str) -> Optional[timedelta]:
    """Parse an ISO 8601 duration string (e.g., 'PT30M', 'PT1H30M')."""
    if not duration_str:
        return None
    
    # Pattern: PT[nH][nM][nS]
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str, re.IGNORECASE)
    if not match:
        # Try parsing simple "30 min" format
        simple_match = re.match(r"(\d+)\s*(min|minutes?|hr|hours?)", duration_str, re.IGNORECASE)
        if simple_match:
            value = int(simple_match.group(1))
            unit = simple_match.group(2).lower()
            if unit.startswith("h"):
                return timedelta(hours=value)
            return timedelta(minutes=value)
        return None
    
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def parse_servings(yields_value) -> tuple[Optional[str], int]:
    """Parse yields/servings from various formats."""
    if yields_value is None:
        return None, 1
    
    if isinstance(yields_value, int):
        return str(yields_value), yields_value
    
    if isinstance(yields_value, list):
        # Format: [4, 'Serves 4']
        servings = yields_value[0] if yields_value else 1
        yields_str = yields_value[1] if len(yields_value) > 1 else str(yields_value[0])
        return yields_str, int(servings)
    
    if isinstance(yields_value, str):
        # Try to extract a number
        match = re.search(r"(\d+)", yields_value)
        servings = int(match.group(1)) if match else 1
        return yields_value, servings
    
    return str(yields_value), 1


def infer_meal_types(name: str, tags: list[str], ingredients: list[Ingredient]) -> list[MealType]:
    """Infer meal types from recipe name, tags, and ingredients."""
    meal_types = []
    
    name_lower = name.lower()
    tags_lower = [t.lower() for t in tags]
    
    # Breakfast indicators
    breakfast_keywords = [
        "breakfast", "morning", "brunch", "egg", "pancake", "waffle",
        "omelette", "frittata", "hash", "muffin", "bagel", "toast"
    ]
    if any(kw in name_lower for kw in breakfast_keywords) or "breakfast" in tags_lower:
        meal_types.append(MealType.BREAKFAST)
    
    # Lunch indicators
    lunch_keywords = ["sandwich", "wrap", "salad", "lunch", "sub", "hoagie"]
    if any(kw in name_lower for kw in lunch_keywords) or "lunch" in tags_lower:
        meal_types.append(MealType.LUNCH)
    
    # Dinner indicators
    dinner_keywords = [
        "dinner", "roast", "stew", "braise", "curry", "pasta", "rice bowl",
        "stir fry", "casserole"
    ]
    if any(kw in name_lower for kw in dinner_keywords) or "dinner" in tags_lower:
        meal_types.append(MealType.DINNER)
    
    # Snack indicators
    snack_keywords = ["snack", "bite", "chip", "dip", "cracker", "popcorn"]
    if any(kw in name_lower for kw in snack_keywords) or "snack" in tags_lower:
        meal_types.append(MealType.SNACK)
    
    # Dessert indicators
    dessert_keywords = [
        "dessert", "cake", "cookie", "pie", "pudding", "ice cream",
        "chocolate", "sweet", "brownie", "cobbler"
    ]
    if any(kw in name_lower for kw in dessert_keywords) or "dessert" in tags_lower:
        meal_types.append(MealType.DESSERT)
    
    # Default to lunch/dinner if no match
    if not meal_types:
        meal_types = [MealType.LUNCH, MealType.DINNER]
    
    return meal_types


def infer_flavor_profiles(name: str, ingredients: list[Ingredient]) -> list[FlavorProfile]:
    """Infer flavor profiles from recipe name and ingredients."""
    profiles = []
    
    name_lower = name.lower()
    ingredient_names = " ".join(ing.name.lower() for ing in ingredients)
    combined = f"{name_lower} {ingredient_names}"
    
    # Spicy indicators
    spicy_keywords = [
        "spicy", "hot", "chili", "chile", "jalapeño", "cayenne", "sriracha",
        "gochujang", "harissa", "buffalo", "pepper flake", "chipotle"
    ]
    if any(kw in combined for kw in spicy_keywords):
        profiles.append(FlavorProfile.SPICY)
    
    # Savory/umami indicators
    savory_keywords = [
        "bacon", "soy sauce", "miso", "mushroom", "parmesan", "anchov",
        "worcestershire", "fish sauce", "beef", "pork", "umami"
    ]
    if any(kw in combined for kw in savory_keywords):
        profiles.append(FlavorProfile.SAVORY)
        profiles.append(FlavorProfile.UMAMI)
    
    # Sweet indicators
    sweet_keywords = [
        "sweet", "honey", "maple", "sugar", "caramel", "chocolate",
        "fruit", "berry", "dessert"
    ]
    if any(kw in combined for kw in sweet_keywords):
        profiles.append(FlavorProfile.SWEET)
    
    # Sour/tangy indicators
    sour_keywords = ["lemon", "lime", "vinegar", "pickle", "sour", "citrus", "tamarind"]
    if any(kw in combined for kw in sour_keywords):
        profiles.append(FlavorProfile.SOUR)
    
    # Fresh indicators
    fresh_keywords = [
        "fresh", "herb", "basil", "cilantro", "mint", "salad", "cucumber",
        "tomato", "avocado"
    ]
    if any(kw in combined for kw in fresh_keywords):
        profiles.append(FlavorProfile.FRESH)
    
    # Rich indicators
    rich_keywords = [
        "cream", "butter", "cheese", "rich", "braised", "short rib",
        "carbonara", "alfredo"
    ]
    if any(kw in combined for kw in rich_keywords):
        profiles.append(FlavorProfile.RICH)
    
    # Light indicators
    light_keywords = ["light", "low calorie", "salad", "grilled", "steamed", "poached"]
    if any(kw in combined for kw in light_keywords):
        profiles.append(FlavorProfile.LIGHT)
    
    return list(set(profiles))


def infer_meal_prep_capability(
    name: str, 
    total_time: Optional[timedelta],
    ingredients: list[Ingredient]
) -> tuple[bool, int]:
    """Infer if a recipe can be meal prepped and for how many days."""
    name_lower = name.lower()
    
    # Recipes that typically meal prep well
    meal_prep_keywords = [
        "make ahead", "prep", "batch", "stew", "braise", "curry",
        "chili", "soup", "casserole", "bowl", "burrito"
    ]
    
    # Recipes that don't meal prep well
    no_prep_keywords = [
        "sandwich", "salad", "fresh", "raw", "crispy", "fried",
        "scrambled", "omelette"
    ]
    
    can_prep = any(kw in name_lower for kw in meal_prep_keywords)
    cant_prep = any(kw in name_lower for kw in no_prep_keywords)
    
    if cant_prep and not can_prep:
        return False, 0
    
    # Default: longer cook time usually means it preps well
    if total_time and total_time.total_seconds() > 45 * 60:
        return True, 3
    
    if can_prep:
        return True, 3
    
    return False, 0


def parse_recipe_file(file_path: Path) -> Optional[Recipe]:
    """Parse a markdown recipe file into a Recipe object."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Extract frontmatter metadata
    metadata = post.metadata
    content = post.content
    
    # Get name from H1 heading or filename
    name_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    name = name_match.group(1) if name_match else file_path.stem
    
    # Parse macros
    macros = Macros(
        calories=float(metadata.get("calories", 0) or 0),
        protein=float(metadata.get("protein", 0) or 0),
        carbs=float(metadata.get("carbs", 0) or 0),
        fat=float(metadata.get("fat", 0) or 0),
    )
    
    # Parse time
    time_str = metadata.get("time", "")
    total_time = parse_iso_duration(time_str) if time_str else None
    
    # Parse yields/servings
    yields_str, servings = parse_servings(metadata.get("yields"))
    
    # Parse ingredients section
    ingredients = []
    ing_match = re.search(r"## Ingredients\n(.*?)(?=## |$)", content, re.DOTALL)
    if ing_match:
        ing_section = ing_match.group(1)
        for line in ing_section.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                ingredient = Ingredient.parse(line)
                if ingredient.name:
                    ingredients.append(ingredient)
    
    # Parse instructions section
    instructions = []
    inst_match = re.search(r"## Instructions\n(.*?)(?=## |$)", content, re.DOTALL)
    if inst_match:
        inst_section = inst_match.group(1)
        # Split by numbered steps or paragraphs
        steps = re.split(r"\n\d+\.\s+", inst_section)
        for step in steps:
            step = step.strip()
            if step:
                instructions.append(step)
    
    # Get tags
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    
    # Infer meal types and flavor profiles
    meal_types = infer_meal_types(name, tags, ingredients)
    flavor_profiles = infer_flavor_profiles(name, ingredients)
    can_prep, prep_days = infer_meal_prep_capability(name, total_time, ingredients)
    
    return Recipe(
        name=name,
        source_path=file_path,
        source_url=metadata.get("source"),
        total_time=total_time,
        yields=yields_str,
        servings=servings,
        macros=macros,
        ingredients=ingredients,
        instructions=instructions,
        tags=tags,
        flavor_profiles=flavor_profiles,
        meal_types=meal_types,
        can_meal_prep=can_prep,
        prep_ahead_days=prep_days,
    )


class RecipeLibrary:
    """A collection of recipes loaded from a directory."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.recipes: dict[str, Recipe] = {}
        self._load_recipes()
    
    def _load_recipes(self):
        """Load all recipes from the base path."""
        if not self.base_path.exists():
            print(f"Warning: Recipe path does not exist: {self.base_path}")
            return
        
        for md_file in self.base_path.rglob("*.md"):
            recipe = parse_recipe_file(md_file)
            if recipe:
                self.recipes[recipe.name] = recipe
        
        print(f"Loaded {len(self.recipes)} recipes from {self.base_path}")
    
    def get_recipe(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self.recipes.get(name)
    
    def search(
        self,
        query: Optional[str] = None,
        meal_types: Optional[list[MealType]] = None,
        moods: Optional[list[str]] = None,
        max_time_minutes: Optional[int] = None,
        min_protein: Optional[float] = None,
        max_calories: Optional[float] = None,
        ingredients_include: Optional[list[str]] = None,
        ingredients_exclude: Optional[list[str]] = None,
    ) -> list[Recipe]:
        """Search recipes with various filters."""
        results = list(self.recipes.values())
        
        if query:
            query_lower = query.lower()
            results = [r for r in results if query_lower in r.name.lower()]
        
        if meal_types:
            results = [r for r in results if any(mt in r.meal_types for mt in meal_types)]
        
        if moods:
            results = [r for r in results if r.matches_mood(moods)]
        
        if max_time_minutes:
            results = [
                r for r in results
                if r.total_time is None or r.total_time.total_seconds() <= max_time_minutes * 60
            ]
        
        if min_protein is not None:
            results = [r for r in results if r.macros.protein >= min_protein]
        
        if max_calories is not None:
            results = [r for r in results if r.macros.calories <= max_calories]
        
        if ingredients_include:
            results = [
                r for r in results
                if any(
                    inc.lower() in ing_name
                    for inc in ingredients_include
                    for ing_name in r.get_ingredient_names()
                )
            ]
        
        if ingredients_exclude:
            results = [
                r for r in results
                if not any(
                    exc.lower() in ing_name
                    for exc in ingredients_exclude
                    for ing_name in r.get_ingredient_names()
                )
            ]
        
        return results
    
    def get_high_protein_recipes(self, min_protein: float = 30) -> list[Recipe]:
        """Get recipes with high protein content."""
        return [r for r in self.recipes.values() if r.macros.protein >= min_protein]
    
    def get_quick_recipes(self, max_minutes: int = 30) -> list[Recipe]:
        """Get recipes that can be made quickly."""
        return [
            r for r in self.recipes.values()
            if r.total_time and r.total_time.total_seconds() <= max_minutes * 60
        ]
    
    def get_meal_prep_recipes(self) -> list[Recipe]:
        """Get recipes suitable for meal prep."""
        return [r for r in self.recipes.values() if r.can_meal_prep]
    
    def get_by_ingredient(self, ingredient: str) -> list[Recipe]:
        """Find recipes containing a specific ingredient."""
        ingredient_lower = ingredient.lower()
        return [
            r for r in self.recipes.values()
            if any(ingredient_lower in ing_name for ing_name in r.get_ingredient_names())
        ]
    
    def get_stats(self) -> dict:
        """Get statistics about the recipe library."""
        recipes = list(self.recipes.values())
        if not recipes:
            return {"total": 0}
        
        proteins = [r.macros.protein for r in recipes if r.macros.protein > 0]
        calories = [r.macros.calories for r in recipes if r.macros.calories > 0]
        
        return {
            "total": len(recipes),
            "with_macros": len([r for r in recipes if r.macros.calories > 0]),
            "avg_protein": sum(proteins) / len(proteins) if proteins else 0,
            "avg_calories": sum(calories) / len(calories) if calories else 0,
            "meal_prep_suitable": len([r for r in recipes if r.can_meal_prep]),
        }
