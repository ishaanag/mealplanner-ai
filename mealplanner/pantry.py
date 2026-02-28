"""Pantry inventory management system."""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import PantryItem, Ingredient


# Ingredient aliases for better matching
# Maps variations -> canonical name
INGREDIENT_ALIASES = {
    # Oils
    "olive oil": "olive oil",
    "extra-virgin olive oil": "olive oil",
    "extra virgin olive oil": "olive oil",
    "evoo": "olive oil",
    "vegetable oil": "olive oil",  # Can substitute
    "canola oil": "olive oil",
    "neutral oil": "olive oil",
    "cooking oil": "olive oil",
    "oil": "olive oil",
    
    # Garlic variations
    "garlic clove": "garlic",
    "garlic cloves": "garlic",
    "minced garlic": "garlic",
    "crushed garlic": "garlic",
    "garlic clove minced": "garlic",
    "garlic peeled": "garlic",
    "fresh garlic": "garlic",
    
    # Onion variations
    "yellow onion": "onions",
    "white onion": "onions",
    "onion": "onions",
    "red onion": "onions",
    "chopped onion": "onions",
    "diced onion": "onions",
    
    # Pepper variations
    "black pepper": "pepper",
    "ground black pepper": "pepper",
    "freshly ground black pepper": "pepper",
    "freshly ground pepper": "pepper",
    "white pepper": "white pepper",
    "ground white pepper": "white pepper",
    
    # Salt variations
    "kosher salt": "salt",
    "sea salt": "salt",
    "table salt": "salt",
    "flaky salt": "salt",
    "fine salt": "salt",
    
    # Cumin variations
    "ground cumin": "cumin powder",
    "cumin": "cumin powder",
    "cumin seed": "cumin seeds",
    
    # Coriander variations
    "ground coriander": "coriander powder",
    "coriander": "coriander powder",
    "coriander seed": "coriander seeds",
    
    # Other spices
    "ground cinnamon": "cinnamon",
    "ground turmeric": "turmeric",
    "turmeric powder": "turmeric",
    "red pepper flakes": "chilli flakes",
    "crushed red pepper": "chilli flakes",
    "chili flakes": "chilli flakes",
    "red chili flakes": "chilli flakes",
    "cayenne": "cayenne pepper",
    "ground cayenne": "cayenne pepper",
    "paprika": "paprika",
    "sweet paprika": "paprika",
    "smoked paprika": "smoked paprika",
    "oregano": "oregano",
    "dried oregano": "oregano",
    "bay leaf": "bay leaves",
    "bay leaves": "bay leaves",
    "thyme": "fresh thyme",
    "fresh thyme": "fresh thyme",
    "dried thyme": "fresh thyme",
    
    # Sesame
    "sesame seeds": "roasted sesame seed",
    "sesame seed": "roasted sesame seed",
    "toasted sesame seeds": "roasted sesame seed",
    "white sesame seeds": "roasted sesame seed",
    
    # Soy sauces
    "soy sauce": "light soy sauce",
    "low-sodium soy sauce": "light soy sauce",
    "shoyu": "light soy sauce",
    
    # Vinegars
    "rice wine vinegar": "rice vinegar",
    "rice vinegar": "rice vinegar",
    "balsamic": "balsamic vinegar",
    "balsamic vinegar": "balsamic vinegar",
    "apple cider vinegar": "apple cider vinegar",
    "white vinegar": "rice vinegar",
    
    # Dairy
    "unsalted butter": "butter",
    "salted butter": "butter",
    "heavy cream": "table cream",
    "heavy whipping cream": "table cream",
    "whipping cream": "table cream",
    
    # Eggs
    "egg": "egg",
    "large egg": "egg",
    "large eggs": "egg",
    "eggs": "egg",
    
    # Pastes and sauces
    "fish sauce": "fish sauce",
    "thai fish sauce": "fish sauce",
    "oyster sauce": "oyster sauce",
    "hoisin sauce": "hoisin sauce",
    "gochujang": "gochujang",
    "korean chili paste": "gochujang",
    "doenjang": "doenjang",
    "korean soybean paste": "doenjang",
    "miso paste": "doenjang",
    "curry paste": "green curry paste",
    "thai curry paste": "green curry paste",
    
    # Rice
    "rice": "jasmine rice",
    "white rice": "jasmine rice",
    "long grain rice": "jasmine rice",
    "basmati": "basmati rice",
    "steamed rice": "jasmine rice",
    "steamed white rice": "jasmine rice",
    "cooked rice": "jasmine rice",
    
    # Noodles
    "pasta": "penne",
    "dried pasta": "penne",
    
    # Ginger
    "fresh ginger": "ginger",
    "ginger root": "ginger",
    "minced ginger": "ginger",
    "piece ginger": "ginger",
    
    # Lime/Lemon
    "lime juice": "lime",
    "fresh lime juice": "lime",
    "lemon juice": "lemon juice",
    "fresh lemon juice": "lemon juice",
    
    # Coconut
    "coconut milk": "coconut milk",
    "full-fat coconut milk": "coconut milk",
    "lite coconut milk": "coconut milk",
    
    # Sugar
    "sugar": "white sugar",
    "granulated sugar": "white sugar",
    "brown sugar": "brown sugar",
    "light brown sugar": "brown sugar",
    "dark brown sugar": "brown sugar",
    
    # Flour
    "flour": "all-purpose flour",
    "all-purpose flour": "all-purpose flour",
    "ap flour": "all-purpose flour",
    
    # Beans
    "chickpeas": "chickpeas",
    "garbanzo beans": "chickpeas",
    "canned chickpeas": "chickpeas",
    "kidney beans": "white kidney beans",
    "white beans": "white kidney beans",
    "cannellini beans": "white kidney beans",
    
    # Tomatoes - map to diced tomatoes in pantry
    "tomatoes": "diced tomatoes",
    "crushed tomatoes": "diced tomatoes",
    "canned tomatoes": "diced tomatoes",
    "can tomatoes": "diced tomatoes",
    "tomato": "tomato",
    
    # Chicken/broth
    "chicken broth": "chicken broth",
    "chicken stock": "chicken broth",
    
    # Scallions
    "scallion": "scallions",
    "scallions": "scallions",
    "green onion": "scallions",
    "green onions": "scallions",
    
    # Cilantro
    "cilantro": "cilantro",
    "fresh cilantro": "cilantro",
    "coriander leaves": "cilantro",
    
    # Parsley
    "parsley": "parsley",
    "fresh parsley": "parsley",
    
    # Carrots
    "carrot": "carrots",
    "carrots": "carrots",
    
    # Celery
    "celery": "celery",
    "celery rib": "celery",
    "celery ribs": "celery",
    
    # Star anise
    "star anise": "star anise",
    "anise pod": "star anise",
    "star anise pod": "star anise",
    
    # Cinnamon
    "cinnamon stick": "cinnamon",
    "cinnamon sticks": "cinnamon",
}


# Category mappings for ingredients
CATEGORY_KEYWORDS = {
    "protein": [
        "chicken", "beef", "pork", "turkey", "lamb", "fish", "salmon", "tuna",
        "shrimp", "tofu", "tempeh", "egg", "bacon", "sausage", "ground"
    ],
    "dairy": [
        "milk", "cream", "cheese", "yogurt", "butter", "sour cream",
        "mozzarella", "parmesan", "cheddar", "feta", "ricotta"
    ],
    "produce": [
        "onion", "garlic", "tomato", "pepper", "lettuce", "spinach", "kale",
        "broccoli", "carrot", "celery", "potato", "mushroom", "zucchini",
        "cucumber", "avocado", "lemon", "lime", "ginger", "cilantro", "basil"
    ],
    "grains": [
        "rice", "pasta", "bread", "tortilla", "noodle", "quinoa", "oat",
        "flour", "panko", "breadcrumb", "bagel", "bun"
    ],
    "canned": [
        "tomato paste", "crushed tomato", "bean", "chickpea", "coconut milk",
        "broth", "stock"
    ],
    "condiments": [
        "soy sauce", "hot sauce", "sriracha", "ketchup", "mustard", "mayo",
        "vinegar", "oil", "honey", "maple", "salsa", "gochujang", "harissa"
    ],
    "spices": [
        "salt", "pepper", "cumin", "paprika", "cayenne", "oregano", "thyme",
        "basil", "garlic powder", "onion powder", "chili powder", "turmeric",
        "cinnamon", "curry"
    ],
    "frozen": [
        "frozen", "ice cream"
    ],
}


def categorize_ingredient(name: str) -> str:
    """Determine the category of an ingredient."""
    name_lower = name.lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    
    return "other"


class Pantry:
    """Manages pantry inventory."""
    
    def __init__(self, pantry_path: Path):
        self.pantry_path = pantry_path
        self.items: dict[str, PantryItem] = {}
        self._load_pantry()
    
    def _load_pantry(self):
        """Load pantry from markdown file."""
        if not self.pantry_path.exists():
            print(f"Pantry file not found: {self.pantry_path}")
            print("Creating empty pantry...")
            self._save_pantry()
            return
        
        with open(self.pantry_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        current_category = "other"
        skip_section = False
        
        # Sections to skip (not actual pantry categories)
        skip_categories = {"format guide", "format", "guide", "notes", "instructions"}
        
        for line in content.split("\n"):
            line = line.strip()
            
            # Check for category headers
            if line.startswith("## "):
                current_category = line[3:].strip().lower()
                skip_section = current_category in skip_categories
                continue
            
            # Skip non-pantry sections
            if skip_section:
                continue
            
            # Parse item lines
            if line.startswith("- "):
                item = self._parse_pantry_line(line, current_category)
                if item:
                    self.items[item.name.lower()] = item
        
        print(f"Loaded {len(self.items)} pantry items")
    
    def _parse_pantry_line(self, line: str, category: str) -> Optional[PantryItem]:
        """Parse a pantry item line."""
        # Remove checkbox and leading dash
        line = re.sub(r"^-\s*(\[.\])?\s*", "", line).strip()
        
        if not line:
            return None
        
        # Try to parse quantity: "Chicken breast, 2 lbs" or "2 lbs chicken breast"
        quantity = None
        unit = None
        name = line
        expiration = None
        
        # Check for expiration date at end: "item (expires: 2024-03-15)"
        exp_match = re.search(r"\(expires?:\s*(\d{4}-\d{2}-\d{2})\)", line)
        if exp_match:
            expiration = exp_match.group(1)
            line = re.sub(r"\s*\(expires?:\s*\d{4}-\d{2}-\d{2}\)", "", line)
        
        # Pattern: "name, quantity unit"
        match = re.match(r"^(.+?),\s*([\d.]+)\s*(\w+)?\s*$", line)
        if match:
            name = match.group(1).strip()
            quantity = float(match.group(2))
            unit = match.group(3)
        else:
            # Pattern: "quantity unit name"
            match = re.match(r"^([\d.]+)\s*(\w+)?\s+(.+)$", line)
            if match:
                quantity = float(match.group(1))
                unit = match.group(2)
                name = match.group(3).strip()
            else:
                name = line
        
        return PantryItem(
            name=name,
            quantity=quantity,
            unit=unit,
            category=category,
            expiration_date=expiration,
        )
    
    def _save_pantry(self):
        """Save pantry to markdown file."""
        lines = [
            "---",
            "tags: [pantry, inventory]",
            f"updated: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "",
            "# Pantry Inventory",
            "",
            "Track what you have on hand. Format: `- Item name, quantity unit`",
            "",
        ]
        
        # Group items by category
        by_category: dict[str, list[PantryItem]] = {}
        for item in self.items.values():
            cat = item.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        # Define category order
        category_order = [
            "protein", "dairy", "produce", "grains", "canned",
            "condiments", "spices", "frozen", "other"
        ]
        
        for category in category_order:
            if category in by_category:
                items = by_category[category]
                lines.append(f"## {category.title()}")
                lines.append("")
                for item in sorted(items, key=lambda x: x.name):
                    line = f"- {item.name}"
                    if item.quantity:
                        line += f", {item.quantity}"
                        if item.unit:
                            line += f" {item.unit}"
                    if item.expiration_date:
                        line += f" (expires: {item.expiration_date})"
                    lines.append(line)
                lines.append("")
        
        with open(self.pantry_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def add_item(
        self,
        name: str,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        category: Optional[str] = None,
        expiration_date: Optional[str] = None,
    ) -> PantryItem:
        """Add or update a pantry item."""
        if category is None:
            category = categorize_ingredient(name)
        
        key = name.lower()
        
        if key in self.items:
            # Update existing item
            existing = self.items[key]
            if quantity is not None:
                existing.quantity = (existing.quantity or 0) + quantity
            if expiration_date:
                existing.expiration_date = expiration_date
        else:
            # Add new item
            self.items[key] = PantryItem(
                name=name,
                quantity=quantity,
                unit=unit,
                category=category,
                expiration_date=expiration_date,
            )
        
        self._save_pantry()
        return self.items[key]
    
    def remove_item(self, name: str, quantity: Optional[float] = None) -> bool:
        """Remove or reduce a pantry item."""
        key = name.lower()
        
        if key not in self.items:
            return False
        
        if quantity is None:
            # Remove entirely
            del self.items[key]
        else:
            # Reduce quantity
            item = self.items[key]
            if item.quantity is not None:
                item.quantity -= quantity
                if item.quantity <= 0:
                    del self.items[key]
        
        self._save_pantry()
        return True
    
    def _normalize_ingredient(self, name: str) -> str:
        """Normalize ingredient name using aliases."""
        name_lower = name.lower().strip()
        
        # Check direct alias match
        if name_lower in INGREDIENT_ALIASES:
            return INGREDIENT_ALIASES[name_lower]
        
        # Remove common prefixes/suffixes
        prefixes = ["fresh ", "dried ", "ground ", "minced ", "chopped ", "sliced "]
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                stripped = name_lower[len(prefix):]
                if stripped in INGREDIENT_ALIASES:
                    return INGREDIENT_ALIASES[stripped]
                name_lower = stripped
        
        return name_lower
    
    def has_item(self, name: str) -> bool:
        """Check if pantry has an item (fuzzy match with aliases)."""
        name_lower = self._normalize_ingredient(name)
        
        # Exact match
        if name_lower in self.items:
            return True
        
        # Check if any pantry item matches via alias
        for alias, canonical in INGREDIENT_ALIASES.items():
            if name_lower == alias or name_lower == canonical:
                if canonical in self.items:
                    return True
        
        # Fuzzy match
        for key in self.items:
            if name_lower in key or key in name_lower:
                return True
        
        return False
    
    def get_item(self, name: str) -> Optional[PantryItem]:
        """Get a pantry item by name (fuzzy match with aliases)."""
        name_lower = self._normalize_ingredient(name)
        
        # Exact match first
        if name_lower in self.items:
            return self.items[name_lower]
        
        # Check via alias
        for alias, canonical in INGREDIENT_ALIASES.items():
            if name_lower == alias or name_lower == canonical:
                if canonical in self.items:
                    return self.items[canonical]
        
        # Fuzzy match
        for key, item in self.items.items():
            if name_lower in key or key in name_lower:
                return item
        
        return None
    
    def check_ingredients(
        self, 
        ingredients: list[Ingredient]
    ) -> tuple[list[Ingredient], list[Ingredient]]:
        """
        Check which ingredients are in pantry.
        Returns (in_pantry, not_in_pantry).
        """
        in_pantry = []
        not_in_pantry = []
        
        for ing in ingredients:
            if self.has_item(ing.name):
                in_pantry.append(ing)
            else:
                not_in_pantry.append(ing)
        
        return in_pantry, not_in_pantry
    
    def get_expiring_soon(self, days: int = 7) -> list[PantryItem]:
        """Get items expiring within the specified days."""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() + timedelta(days=days)
        expiring = []
        
        for item in self.items.values():
            if item.expiration_date:
                try:
                    exp_date = datetime.strptime(item.expiration_date, "%Y-%m-%d")
                    if exp_date <= cutoff:
                        expiring.append(item)
                except ValueError:
                    pass
        
        return sorted(expiring, key=lambda x: x.expiration_date or "")
    
    def get_by_category(self, category: str) -> list[PantryItem]:
        """Get all items in a category."""
        return [item for item in self.items.values() if item.category == category]
    
    def to_markdown(self) -> str:
        """Export pantry to markdown format."""
        lines = ["# Pantry Inventory\n"]
        
        by_category: dict[str, list[PantryItem]] = {}
        for item in self.items.values():
            cat = item.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)
        
        for category, items in sorted(by_category.items()):
            lines.append(f"\n## {category.title()}\n")
            for item in sorted(items, key=lambda x: x.name):
                line = f"- {item.name}"
                if item.quantity:
                    line += f", {item.quantity}"
                    if item.unit:
                        line += f" {item.unit}"
                lines.append(line)
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get statistics about the pantry."""
        by_category: dict[str, int] = {}
        for item in self.items.values():
            cat = item.category or "other"
            by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "total_items": len(self.items),
            "by_category": by_category,
            "expiring_soon": len(self.get_expiring_soon()),
        }


def create_sample_pantry(pantry_path: Path):
    """Create a sample pantry file with common items."""
    sample_content = """---
tags: [pantry, inventory]
updated: 2026-02-28
---

# Pantry Inventory

Track what you have on hand. Format: `- Item name, quantity unit`

## Protein
- Chicken breast, 2 lbs
- Ground beef 93/7, 1 lb
- Eggs, 12
- Bacon, 1 lb

## Dairy
- Milk, 1 gallon
- Cheddar cheese, 8 oz
- Greek yogurt, 32 oz
- Butter, 1 lb
- Sour cream, 8 oz

## Produce
- Onions, 3
- Garlic, 1 head
- Tomatoes, 4
- Bell peppers, 2
- Spinach, 1 bag
- Lemons, 3
- Limes, 2
- Cilantro, 1 bunch
- Avocados, 2

## Grains
- Rice, 5 lbs
- Pasta, 2 lbs
- Flour tortillas, 10
- Bread, 1 loaf
- Panko breadcrumbs, 1 box

## Canned
- Crushed tomatoes, 2 cans
- Black beans, 2 cans
- Chicken broth, 4 cups
- Coconut milk, 1 can

## Condiments
- Soy sauce, 1 bottle
- Hot sauce, 1 bottle
- Olive oil, 1 bottle
- Vegetable oil, 1 bottle
- Honey, 1 jar
- Mustard, 1 bottle
- Mayo, 1 jar

## Spices
- Salt
- Black pepper
- Cumin
- Paprika
- Cayenne pepper
- Garlic powder
- Onion powder
- Oregano
- Chili powder
- Cinnamon

## Frozen
- Frozen peas, 1 bag
- Frozen corn, 1 bag
"""
    
    with open(pantry_path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print(f"Created sample pantry at {pantry_path}")
