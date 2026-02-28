"""Shopping list generation from meal plans."""

from collections import defaultdict
from typing import Optional
import json

from .models import (
    ShoppingList, ShoppingListItem, MealPlan, Ingredient,
)
from .pantry import Pantry, categorize_ingredient


# Unit normalization mappings
UNIT_CONVERSIONS = {
    # Volume
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "cup": "cups",
    "ounce": "oz",
    "ounces": "oz",
    "pound": "lbs",
    "pounds": "lbs",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "milliliter": "ml",
    "milliliters": "ml",
    "liter": "L",
    "liters": "L",
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Normalize a unit to a standard form."""
    if unit is None:
        return None
    return UNIT_CONVERSIONS.get(unit.lower(), unit.lower())


def is_valid_unit(unit: Optional[str]) -> bool:
    """Check if a unit is valid (not an ingredient name)."""
    if not unit:
        return False
    unit_lower = unit.lower()
    # These are ingredient names or descriptors, not units
    invalid_units = {
        "cinnamon", "star", "poblano", "scallion", "scallions", 
        "serrano", "garlic", "ginger", "onion", "anise",
        "parsley", "cilantro", "thyme", "rosemary", "basil",
        "thin", "thick", "large", "small", "medium", "slices",
        "inch", "inches",
    }
    if unit_lower in invalid_units:
        return False
    # Units should be short
    if len(unit) > 10:
        return False
    return True


def format_quantity(qty: Optional[float], unit: Optional[str]) -> str:
    """Format quantity and unit for display."""
    if not qty:
        return ""
    
    # Check if unit is valid
    if unit and is_valid_unit(unit):
        return f"{qty:.1f} {unit}"
    else:
        # Just show quantity without unit
        return f"{qty:.0f}" if qty == int(qty) else f"{qty:.1f}"


def extract_base_ingredient(name: str) -> str:
    """Extract the base ingredient name, stripping all prep instructions."""
    import re
    
    original = name
    name = name.lower().strip()
    
    # Remove parenthetical notes
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove leading quantity/size patterns like "1½ pounds", "½-inch", "2½ tablespoons", etc.
    name = re.sub(r'^[\d½¼¾⅓⅔⅛]+\s*(pounds?|lbs?|oz|ounces?|cups?|inch|inches|tbsp|tablespoons?|tsp|teaspoons?|g|kg|ml|l)\s*', '', name, flags=re.IGNORECASE)
    
    # Remove leading fractional amounts without units
    name = re.sub(r'^[\d½¼¾⅓⅔⅛]+\s+', '', name)
    
    # Remove common prep phrases (order matters - longer phrases first)
    prep_phrases = [
        # Multi-word prep instructions - catch everything after these
        r',?\s*white and green parts?.*$',
        r',?\s*green parts?.*$',
        r',?\s*roots?\s*trimmed.*$',
        r',?\s*peeled and.*$',
        r',?\s*trimmed and.*$',
        r',?\s*cut into.*$',
        r',?\s*husks and.*$',
        r',?\s*stems? removed.*$',
        r',?\s*halved .*$',
        r',?\s*quartered .*$',
        r',?\s*plus .*$',
        r',?\s*unpeeled.*$',
        r',?\s*[,\s]*and ¾-inch.*$',
        r',?\s*[,\s]*and ½-inch.*$',
        r',?\s*½-inch.*$',
        r',?\s*¾-inch.*$',
        r',?\s*divided.*$',
        r',?\s*for serving.*$',
        r',?\s*for garnish.*$',
        r',?\s*to taste.*$',
        r',?\s*as needed.*$',
        r',?\s*at room temperature.*$',
        r',?\s*room temperature.*$',
        r',?\s*fine\s*$',  # trailing "fine"
        r',?\s*and coarse.*$',  # "and coarsely chopped"
        r',?\s*deveined.*$',  # shrimp prep
        r',?\s*tails? removed.*$',  # shrimp prep
        r',?\s*shells? removed.*$',  # shrimp prep
        r',?\s*peeled\s*$',
        r',?\s*and\s*$',  # trailing "and"
    ]
    
    for pattern in prep_phrases:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Size/prep descriptors
    name = re.sub(r'\d+-inch[\w\s-]*', '', name)
    name = re.sub(r'\d+ inch[\w\s-]*', '', name)
    name = re.sub(r'½-inch[\w\s-]*', '', name)
    name = re.sub(r'¼-inch[\w\s-]*', '', name)
    name = re.sub(r'¾-inch[\w\s-]*', '', name)
        
    # Common prep words
    unit_words = [
        r'\bpiece\b', r'\bpieces\b', r'\bhead\b', r'\bheads\b',
        r'\brib\b', r'\bribs\b', r'\bstick\b', r'\bsticks\b',
        r'\bclove\b', r'\bcloves\b', r'\bsprig\b', r'\bsprigs\b',
        r'\bleaf\b', r'\bleaves\b', r'\bpod\b', r'\bpods\b',
        r'\bcan\b', r'\bcans\b', r'\bbunch\b', r'\bbunches\b',
        r'\broots?\b', r'\bfillet\b', r'\bfillets\b',
    ]
    
    for pattern in unit_words:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Remove single-word descriptors
    descriptors = [
        "fresh", "dried", "ground", "whole", "large", "small", "medium",
        "extra-virgin", "virgin", "organic", "raw", "cooked", "frozen",
        "canned", "diced", "sliced", "chopped", "minced", "grated", "crushed",
        "peeled", "trimmed", "quartered", "halved", "seeded", "stemmed",
        "boneless", "skinless", "bone-in", "skin-on", "thin", "thick",
        "finely", "coarsely", "roughly", "thinly", "crosswise", "lengthwise",
        "shredded", "julienned", "cubed", "torn", "rinsed", "drained",
        "undrained", "packed", "loosely", "lightly", "well", "separated",
        "reserved", "warmed", "melted", "softened", "cold", "hot", "warm",
        "split", "unpeeled", "extra-", "extra", "fermented",
    ]
    
    for desc in descriptors:
        name = re.sub(rf'\b{re.escape(desc)}\b', '', name, flags=re.IGNORECASE)
    
    # Clean up commas and extra spaces
    name = re.sub(r',\s*,+', '', name)  # Remove consecutive commas
    name = re.sub(r',\s*and\s*$', '', name)  # Remove trailing ", and"
    name = re.sub(r'^\s*,\s*', '', name)
    name = re.sub(r'\s*,\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip(' ,')
    
    return name if name else original.lower()


def normalize_ingredient_name(name: str) -> str:
    """Normalize ingredient name for grouping and pantry matching."""
    # First extract the base ingredient
    name = extract_base_ingredient(name)
    
    # Handle common variations - map to canonical names
    variations = {
        "garlic": "garlic",
        "garlic clove": "garlic",
        "clove": "garlic",  # "clove, minced" usually means garlic
        "onion": "onion",
        "onions": "onion",
        "red onion": "onion",
        "white onion": "onion",
        "yellow onion": "onion",
        "egg": "egg",
        "eggs": "egg",
        "tomato": "tomatoes",
        "tomatoes": "tomatoes",
        "crushed tomatoes": "tomatoes",
        "diced tomatoes": "tomatoes",
        "cherry tomatoes": "tomatoes",
        "pepper": "bell pepper",
        "peppers": "bell pepper",
        "bell pepper": "bell pepper",
        "bell peppers": "bell pepper",
        "green bell pepper": "bell pepper",
        "red bell pepper": "bell pepper",
        "ginger": "ginger",
        "ginger root": "ginger",
        "chicken broth": "chicken broth",
        "chicken stock": "chicken broth",
        "vegetable broth": "vegetable broth",
        "vegetable stock": "vegetable broth",
        "soy sauce": "soy sauce",
        "low sodium soy sauce": "soy sauce",
        "scallion": "scallions",
        "scallions": "scallions",
        "green onion": "scallions",
        "green onions": "scallions",
        "cilantro": "cilantro",
        "coriander": "cilantro",
        "fresh cilantro": "cilantro",
        # Chiles
        "poblano": "poblano chiles",
        "poblanos": "poblano chiles",
        "poblano chile": "poblano chiles",
        "poblano chiles": "poblano chiles",
        "chiles": "chiles",
        "serrano": "serrano chiles",
        "serrano chile": "serrano chiles",
        "serrano chiles": "serrano chiles",
        # Star anise
        "star anise": "star anise",
        "anise": "star anise",
        "star anise pod": "star anise",
        # Cinnamon
        "cinnamon": "cinnamon",
        "cinnamon stick": "cinnamon",
        "cinnamon sticks": "cinnamon",
        # Potatoes
        "sweet potato": "sweet potatoes",
        "sweet potatoes": "sweet potatoes",
        "potato": "potatoes",
        "potatoes": "potatoes",
        # Bok choy
        "bok choy": "baby bok choy",
        "baby bok choy": "baby bok choy",
        # Tomatillos
        "tomatillo": "tomatillos",
        "tomatillos": "tomatillos",
        # Chicken
        "chicken thigh": "chicken thighs",
        "chicken thighs": "chicken thighs",
        "chicken breast": "chicken breasts",
        "chicken breasts": "chicken breasts",
        # Wine
        "shaoxing": "shaoxing wine",
        "shaoxing wine": "shaoxing wine",
        # Vinegar
        "white vinegar": "white vinegar",
        "distilled white vinegar": "white vinegar",
        # Seeds
        "pepitas": "pepitas",
        "roasted pepitas": "pepitas",
        "pumpkin seeds": "pepitas",
        # Shrimp
        "shrimp": "shrimp",
        # Black beans
        "black beans": "black beans",
        "fermented black beans": "fermented black beans",
        # Common pantry items
        "sugar": "sugar",
        "cornstarch": "cornstarch",
        "white": "scallions",  # "white and green parts" = scallions
    }
    
    # Check exact match first
    if name in variations:
        return variations[name]
    
    # Check if name contains a key ingredient
    for var, normalized in variations.items():
        if var in name:
            return normalized
    
    return name.strip()


def clean_display_name(normalized_name: str) -> str:
    """Create a clean display name for shopping list."""
    import re
    
    name = normalized_name.strip()
    
    # Handle special cases with explicit mappings
    special_cases = {
        "egg": "Eggs",
        "onion": "Onion",
        "tomatoes": "Tomatoes",
        "bell pepper": "Bell Peppers",
        "scallions": "Scallions",
        "cilantro": "Cilantro",
        "ginger": "Ginger",
        "garlic": "Garlic",
        "chicken broth": "Chicken Broth",
        "soy sauce": "Soy Sauce",
        "star anise": "Star Anise",
        "anise": "Star Anise",
        "cinnamon": "Cinnamon Sticks",
        "stick": "Cinnamon Sticks",  # "Stick" alone usually means cinnamon
        "serrano chile": "Serrano Chiles",
        "serrano chiles": "Serrano Chiles",
        "serrano": "Serrano Chiles",
        "poblano": "Poblano Chiles",
        "poblano chiles": "Poblano Chiles",
        "chiles": "Chiles",
        "tomatillos": "Tomatillos",
        "sweet potatoes": "Sweet Potatoes",
        "sweet potato": "Sweet Potatoes",
        "potatoes": "Potatoes",
        "bok choy": "Baby Bok Choy",
        "baby bok choy": "Baby Bok Choy",
        "chinese wheat noodles": "Chinese Wheat Noodles",
        "wheat noodles": "Chinese Wheat Noodles",
        "roasted pepitas": "Pepitas",
        "pepitas": "Pepitas",
        "chicken thighs": "Chicken Thighs",
        "chicken thigh": "Chicken Thighs",
        "chicken breasts": "Chicken Breasts",
        "chicken breast": "Chicken Breasts",
        "shaoxing wine": "Shaoxing Wine",
        "distilled white vinegar": "White Vinegar",
        "white vinegar": "White Vinegar",
        "ancho chili powder": "Ancho Chili Powder",
        "chili powder": "Chili Powder",
        "bacon": "Bacon",
        "roots trimmed": "Scallions",  # "Roots Trimmed" is from scallions
        "parsley": "Parsley",
        "lemon slices": "Lemons",
        "lemon": "Lemons",
        "lime slices": "Limes",
        "lime": "Limes",
        "halibut fillets": "Halibut",
        "halibut": "Halibut",
        "salmon fillets": "Salmon",
        "salmon": "Salmon",
        "shrimp": "Shrimp",
        "black beans": "Black Beans",
        "fermented black beans": "Fermented Black Beans",
        "sugar": "Sugar",
        "cornstarch": "Cornstarch",
    }
    
    # Check exact match
    if name in special_cases:
        return special_cases[name]
    
    # Check partial match - if the key is contained in the name
    # Sort by length descending to match longer keys first
    for key in sorted(special_cases.keys(), key=len, reverse=True):
        if key in name:
            return special_cases[key]
    
    # Clean up any remaining messy characters
    name = re.sub(r',\s*,+', '', name)
    name = re.sub(r'^\s*,\s*', '', name)
    name = re.sub(r'\s*,\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip(' ,')
    
    # If the name is too short or empty after cleanup, try to extract something
    if len(name) < 2:
        return normalized_name.title()
    
    return name.title()


class ShoppingListGenerator:
    """Generate shopping lists from meal plans."""
    
    def __init__(self, pantry: Optional[Pantry] = None, gemini_api_key: Optional[str] = None):
        self.pantry = pantry
        self.gemini_api_key = gemini_api_key
    
    def _clean_with_llm(self, raw_items: list[dict]) -> list[dict]:
        """Use LLM to clean and consolidate shopping list items."""
        if not self.gemini_api_key:
            return raw_items
            
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            prompt = f"""Clean up this shopping list. Fix ingredient names, merge duplicates, and make it readable.

Rules:
1. Fix malformed names like "But Firm Pear" → "Pears (ripe)" 
2. Merge duplicates: "Chile, 4" + "Chiles, 14" → "Poblano Chiles, 18"
3. Fix "Chopped, 2 onion" → "Onion, 2" (remove prep descriptors)
4. Clean names: remove ALL prep instructions, keep just the ingredient
5. Combine items with same base ingredient (e.g., all poblanos together)
6. Use sensible units: "lbs" for meat/produce weight, "cups" for liquids, count for produce items
7. Round quantities to reasonable numbers (no 0.7 tsp)
8. REMOVE these pantry staples entirely - do NOT include them: water, salt, pepper, cooking oil, vegetable oil, olive oil, butter, soy sauce, fish sauce, sugar, flour (unless large amounts like 2+ cups)
9. Remove any item that looks like a partial phrase or doesn't make sense as a grocery item

Input items (JSON):
{json.dumps(raw_items, indent=2)}

Return ONLY a JSON array with cleaned items. Each item should have:
- "name": Clean ingredient name (e.g., "Chicken Breasts", "Poblano Chiles", "Pears")
- "quantity": Number or null
- "unit": Unit string or null (use standard: lbs, oz, cups, tbsp, tsp, or omit for count items)
- "category": One of: protein, produce, dairy, grains, canned, condiments, spices, frozen, other

Example output format:
[
  {{"name": "Chicken Breasts", "quantity": 4, "unit": "lbs", "category": "protein"}},
  {{"name": "Poblano Chiles", "quantity": 18, "unit": null, "category": "produce"}}
]

Return ONLY valid JSON, no markdown or explanation."""

            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            
            cleaned = json.loads(text)
            return cleaned
            
        except Exception as e:
            print(f"Warning: LLM cleanup failed ({e}), using raw items")
            return raw_items
    
    def generate(self, meal_plan: MealPlan) -> ShoppingList:
        """Generate a shopping list from a meal plan."""
        # Aggregate all ingredients
        aggregated: dict[str, dict] = defaultdict(lambda: {
            "quantity": 0,
            "unit": None,
            "raw_names": set(),
        })
        
        for day in meal_plan.days:
            for meal in day.meals:
                if meal.recipe:
                    for ing in meal.recipe.ingredients:
                        normalized = normalize_ingredient_name(ing.name)
                        
                        if ing.quantity:
                            scaled_qty = ing.quantity * meal.servings
                            
                            # Handle unit conversion/aggregation
                            existing_unit = aggregated[normalized]["unit"]
                            new_unit = normalize_unit(ing.unit)
                            
                            if existing_unit is None or existing_unit == new_unit:
                                aggregated[normalized]["quantity"] += scaled_qty
                                aggregated[normalized]["unit"] = new_unit
                            else:
                                # Different units - keep separate or convert
                                # For now, just add to quantity (simplification)
                                aggregated[normalized]["quantity"] += scaled_qty
                        
                        aggregated[normalized]["raw_names"].add(ing.name)
        
        # Convert to shopping list items
        items = []
        for normalized_name, data in aggregated.items():
            # Check pantry - try multiple variations
            in_pantry = False
            pantry_qty = None
            is_staple = False
            
            if self.pantry:
                # Try normalized name first
                pantry_item = self.pantry.get_item(normalized_name)
                
                # If not found, try each raw name
                if not pantry_item:
                    for raw_name in data["raw_names"]:
                        pantry_item = self.pantry.get_item(raw_name)
                        if pantry_item:
                            break
                
                # Also try base ingredient extraction on raw names
                if not pantry_item:
                    for raw_name in data["raw_names"]:
                        base = extract_base_ingredient(raw_name)
                        pantry_item = self.pantry.get_item(base)
                        if pantry_item:
                            break
                
                if pantry_item:
                    in_pantry = True
                    pantry_qty = pantry_item.quantity
                    is_staple = pantry_item.is_staple  # No quantity = always in stock
            
            # Calculate needed quantity
            # Staples (no quantity) are assumed always available - no need to buy
            needed_qty = data["quantity"]
            if is_staple:
                needed_qty = None  # Staples don't need to be purchased
            elif in_pantry and pantry_qty and needed_qty:
                needed_qty = max(0, needed_qty - pantry_qty)
            
            # Use clean display name instead of messy raw name
            display_name = clean_display_name(normalized_name)
            
            items.append(ShoppingListItem(
                name=display_name,
                quantity=data["quantity"] if data["quantity"] > 0 else None,
                unit=data["unit"],
                category=categorize_ingredient(normalized_name),
                in_pantry=in_pantry,
                pantry_quantity=pantry_qty,
                needed_quantity=needed_qty if needed_qty and needed_qty > 0 else None,
            ))
        
        # Filter to items that need to be bought
        items_to_buy = [i for i in items if not i.in_pantry or (i.needed_quantity and i.needed_quantity > 0)]
        
        # Use LLM to clean up the shopping list
        if items_to_buy:
            raw_items = [
                {
                    "name": i.name,
                    "quantity": i.quantity,
                    "unit": i.unit,
                    "category": i.category,
                }
                for i in items_to_buy
            ]
            
            cleaned = self._clean_with_llm(raw_items)
            
            # Rebuild items from cleaned data
            cleaned_items = []
            for item_data in cleaned:
                cleaned_items.append(ShoppingListItem(
                    name=item_data.get("name", "Unknown"),
                    quantity=item_data.get("quantity"),
                    unit=item_data.get("unit"),
                    category=item_data.get("category", "other"),
                    in_pantry=False,
                    pantry_quantity=None,
                    needed_quantity=item_data.get("quantity"),
                ))
            
            # Add back pantry items that don't need buying
            pantry_only = [i for i in items if i.in_pantry and (not i.needed_quantity or i.needed_quantity <= 0)]
            items = cleaned_items + pantry_only
        
        # Sort by category and name
        items.sort(key=lambda x: (x.category, x.name))
        
        return ShoppingList(items=items)
    
    def generate_from_recipes(
        self,
        recipes: list[tuple],  # List of (Recipe, servings)
    ) -> ShoppingList:
        """Generate a shopping list from a list of recipes with servings."""
        # Create a temporary meal plan
        from models import MealPlan, DayPlan, MealSlot, MealType
        
        meals = []
        for recipe, servings in recipes:
            meals.append(MealSlot(
                day=1,
                meal_type=MealType.DINNER,
                recipe=recipe,
                servings=servings,
            ))
        
        day = DayPlan(day=1, meals=meals)
        plan = MealPlan(days=[day])
        
        return self.generate(plan)
    
    def to_markdown(self, shopping_list: ShoppingList, show_pantry: bool = False) -> str:
        """Convert shopping list to markdown format."""
        lines = ["# 🛒 Shopping List\n"]
        
        by_category = shopping_list.get_by_category()
        
        # Define category display order
        category_order = [
            "protein", "produce", "dairy", "grains", "canned",
            "condiments", "spices", "frozen", "other"
        ]
        
        # Items to buy
        lines.append("## To Buy\n")
        
        for category in category_order:
            if category not in by_category:
                continue
            
            items = [i for i in by_category[category] if not i.in_pantry or i.needed_quantity]
            if not items:
                continue
            
            lines.append(f"### {category.title()}")
            for item in sorted(items, key=lambda x: x.name):
                qty_str = ""
                if item.needed_quantity:
                    qty_str = format_quantity(item.needed_quantity, item.unit)
                elif item.quantity:
                    qty_str = format_quantity(item.quantity, item.unit)
                
                if qty_str:
                    qty_str = f" ({qty_str})"
                
                lines.append(f"- [ ] {item.name}{qty_str}")
            lines.append("")
        
        # Items already in pantry
        if show_pantry:
            pantry_items = [i for i in shopping_list.items if i.in_pantry and not i.needed_quantity]
            if pantry_items:
                lines.append("\n## ✅ Already in Pantry\n")
                for item in sorted(pantry_items, key=lambda x: x.name):
                    lines.append(f"- {item.name}")
        
        return "\n".join(lines)
    
    def to_clean_markdown(self, shopping_list: ShoppingList) -> str:
        """Convert shopping list to a clean, Obsidian-friendly markdown format."""
        from datetime import datetime
        
        lines = [
            "---",
            "tags: [shopping, groceries]",
            f"created: {datetime.now().strftime('%Y-%m-%d')}",
            "---",
            "",
            "# 🛒 Shopping List",
            "",
        ]
        
        by_category = shopping_list.get_by_category()
        
        # Map categories to store sections with emojis
        store_sections = {
            "produce": ("🥬 Produce", 1),
            "protein": ("🥩 Meat & Seafood", 2),
            "dairy": ("🧀 Dairy", 3),
            "frozen": ("❄️ Frozen", 4),
            "grains": ("🍞 Bakery & Grains", 5),
            "canned": ("🥫 Canned & Jarred", 6),
            "condiments": ("🫙 Condiments & Sauces", 7),
            "spices": ("🧂 Spices & Seasonings", 8),
            "other": ("📦 Other", 9),
        }
        
        # Count total items to buy
        total_items = 0
        
        # Sort sections by store order
        sorted_categories = sorted(
            [(cat, info) for cat, info in store_sections.items() if cat in by_category],
            key=lambda x: x[1][1]
        )
        
        for category, (section_name, _) in sorted_categories:
            items = [i for i in by_category[category] if not i.in_pantry or i.needed_quantity]
            if not items:
                continue
            
            lines.append(f"## {section_name}")
            lines.append("")
            
            for item in sorted(items, key=lambda x: x.name):
                # Format quantity nicely
                qty_str = ""
                qty = item.needed_quantity or item.quantity
                if qty and item.unit:
                    # Clean up the quantity display
                    if qty == int(qty):
                        qty_str = f", {int(qty)} {item.unit}"
                    else:
                        qty_str = f", {qty:.1f} {item.unit}"
                elif qty:
                    if qty == int(qty):
                        qty_str = f", {int(qty)}"
                    else:
                        qty_str = f", {qty:.1f}"
                
                lines.append(f"- [ ] {item.name}{qty_str}")
                total_items += 1
            
            lines.append("")
        
        # Add summary at the top after the header
        lines.insert(7, f"**{total_items} items** to buy\n")
        
        return "\n".join(lines)
    
    def to_grouped_text(self, shopping_list: ShoppingList) -> str:
        """Convert to plain text grouped by store section."""
        lines = ["SHOPPING LIST", "=" * 40, ""]
        
        by_category = shopping_list.get_by_category()
        
        # Map categories to store sections
        store_sections = {
            "produce": "🥬 PRODUCE",
            "protein": "🥩 MEAT & SEAFOOD", 
            "dairy": "🧀 DAIRY",
            "grains": "🍞 BAKERY & GRAINS",
            "canned": "🥫 CANNED GOODS",
            "condiments": "🫙 CONDIMENTS",
            "spices": "🧂 SPICES",
            "frozen": "❄️ FROZEN",
            "other": "📦 OTHER",
        }
        
        for category, section_name in store_sections.items():
            if category not in by_category:
                continue
            
            items = [i for i in by_category[category] if not i.in_pantry]
            if not items:
                continue
            
            lines.append(section_name)
            lines.append("-" * 30)
            
            for item in sorted(items, key=lambda x: x.name):
                qty_str = ""
                if item.quantity:
                    qty_str = format_quantity(item.quantity, item.unit)
                    if qty_str:
                        qty_str = f" - {qty_str}"
                
                lines.append(f"  □ {item.name}{qty_str}")
            
            lines.append("")
        
        return "\n".join(lines)
