"""AI-powered meal planning using Google Gemini."""

import json
import re
from pathlib import Path
from typing import Optional
from datetime import timedelta

import google.generativeai as genai

from .models import (
    Recipe, MealPlan, DayPlan, MealSlot, MealType, Macros,
    Ingredient, ShoppingList,
)
from .config import Config, MacroTargets, MealPlanConfig
from .recipe_parser import RecipeLibrary
from .pantry import Pantry
from .macro_tracker import MacroTracker
from .shopping import ShoppingListGenerator


class MealPlanner:
    """AI-powered meal planner using Google Gemini."""
    
    def __init__(
        self,
        config: Config,
        recipe_library: RecipeLibrary,
        pantry: Optional[Pantry] = None,
    ):
        self.config = config
        self.recipe_library = recipe_library
        self.pantry = pantry
        self.macro_tracker = MacroTracker(config.macro_targets)
        self.shopping_generator = ShoppingListGenerator(pantry, gemini_api_key=config.gemini_api_key)
        
        # Initialize Gemini
        if config.gemini_api_key:
            genai.configure(api_key=config.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.5-flash")
        else:
            self.model = None
            print("Warning: No Gemini API key configured. AI features disabled.")
    
    def _build_recipe_context(self, recipes: list[Recipe]) -> str:
        """Build context string describing available recipes."""
        lines = []
        for recipe in recipes:
            time_str = ""
            if recipe.total_time:
                minutes = int(recipe.total_time.total_seconds() / 60)
                time_str = f", {minutes}min"
            
            flavors = ", ".join(p.value for p in recipe.flavor_profiles[:3])
            meals = ", ".join(m.value for m in recipe.meal_types)
            
            lines.append(
                f"- {recipe.name}: {recipe.macros.calories:.0f}cal, "
                f"{recipe.macros.protein:.1f}g protein, {recipe.macros.carbs:.1f}g carbs, "
                f"{recipe.macros.fat:.1f}g fat{time_str} [{meals}] ({flavors})"
            )
        
        return "\n".join(lines)
    
    def _build_pantry_context(self) -> str:
        """Build context string describing pantry contents."""
        if not self.pantry:
            return "No pantry inventory available."
        
        lines = ["Current pantry items:"]
        by_category = {}
        for item in self.pantry.items.values():
            cat = item.category or "other"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item.name)
        
        for cat, items in sorted(by_category.items()):
            lines.append(f"  {cat.title()}: {', '.join(items)}")
        
        return "\n".join(lines)
    
    def _score_recipes_for_selection(self, recipes: list[Recipe], max_recipes: int = 200) -> list[Recipe]:
        """Score and select the best recipes for AI planning.
        
        Scoring considers:
        - Pantry coverage (60%): prioritize recipes using pantry ingredients
        - Ingredient overlap (20%): prefer recipes sharing common ingredients (simpler shopping)
        - Prep time variety (10%): include mix of quick and longer recipes
        - Cuisine diversity (10%): ensure variety across cuisines
        
        Returns a diverse mix of breakfast/lunch/dinner recipes.
        """
        import random
        from collections import Counter
        
        # First pass: count ingredient frequency across all recipes
        ingredient_counts = Counter()
        for recipe in recipes:
            for ing_name in recipe.get_ingredient_names():
                # Normalize to base ingredient
                base = ing_name.lower().split(',')[0].strip()
                # Remove common modifiers
                for word in ['fresh', 'dried', 'chopped', 'minced', 'sliced', 'large', 'small']:
                    base = base.replace(word, '').strip()
                if len(base) > 2:
                    ingredient_counts[base] += 1
        
        # Get the most common ingredients (top 50)
        common_ingredients = set(ing for ing, count in ingredient_counts.most_common(50))
        
        # Track cuisines for diversity scoring
        cuisine_counts = Counter()
        
        scored = []
        for recipe in recipes:
            # 1. Pantry coverage score (0-1): 50% weight
            pantry_score = self._calculate_pantry_coverage(recipe)
            
            # 2. Ingredient overlap score (0-1): 25% weight
            # Higher score if recipe uses commonly-shared ingredients
            recipe_ingredients = set()
            for ing_name in recipe.get_ingredient_names():
                base = ing_name.lower().split(',')[0].strip()
                for word in ['fresh', 'dried', 'chopped', 'minced', 'sliced', 'large', 'small']:
                    base = base.replace(word, '').strip()
                if len(base) > 2:
                    recipe_ingredients.add(base)
            
            overlap_count = len(recipe_ingredients & common_ingredients)
            overlap_score = min(1.0, overlap_count / 8)  # 8+ common ingredients = perfect score
            
            # 3. Prep time score (0-1): 15% weight
            # Good variety: some quick (< 30 min), some medium (30-60), some longer
            if recipe.total_time:
                mins = recipe.total_time.total_seconds() / 60
                if mins <= 30:
                    time_score = 1.0  # Quick recipes are great
                elif mins <= 60:
                    time_score = 0.8  # Medium is good
                else:
                    time_score = 0.5  # Longer still valuable for variety
            else:
                time_score = 0.7  # Unknown time, assume medium
            
            # 4. Cuisine diversity (0-1): 10% weight
            # Extract cuisine from flavor profiles or tags
            cuisines = []
            for fp in recipe.flavor_profiles:
                cuisines.append(fp.value)
            cuisine_key = tuple(sorted(cuisines[:2])) if cuisines else ('general',)
            
            # Score inversely to how often we've seen this cuisine
            # (will be adjusted in second pass)
            cuisine_score = 1.0  # Placeholder, adjusted below
            
            # Combined score (initial, before cuisine adjustment)
            combined = (
                pantry_score * 0.60 +
                overlap_score * 0.20 +
                time_score * 0.10 +
                cuisine_score * 0.10
            )
            
            scored.append({
                'recipe': recipe,
                'combined': combined,
                'pantry': pantry_score,
                'overlap': overlap_score,
                'time': time_score,
                'cuisine_key': cuisine_key,
            })
        
        # Sort by initial score
        scored.sort(key=lambda x: x['combined'], reverse=True)
        
        # Select top recipes with cuisine diversity
        # Use a greedy approach: pick best, but penalize repeating cuisines
        selected = []
        cuisine_selected = Counter()
        
        # Separate by meal type first
        breakfast_pool = [s for s in scored if MealType.BREAKFAST in s['recipe'].meal_types]
        lunch_pool = [s for s in scored if MealType.LUNCH in s['recipe'].meal_types]
        dinner_pool = [s for s in scored if MealType.DINNER in s['recipe'].meal_types]
        
        def select_diverse(pool, count):
            """Select from pool with cuisine diversity."""
            result = []
            local_cuisine_count = Counter()
            
            for item in pool:
                if len(result) >= count:
                    break
                
                # Penalize if we already have many of this cuisine
                cuisine = item['cuisine_key']
                cuisine_penalty = local_cuisine_count[cuisine] * 0.15
                adjusted_score = item['combined'] - cuisine_penalty
                
                # Accept if score still decent or we need variety
                if adjusted_score > 0.2 or len(result) < count // 2:
                    result.append(item['recipe'])
                    local_cuisine_count[cuisine] += 1
            
            return result
        
        # Select from each pool
        breakfast_selected = select_diverse(breakfast_pool, max_recipes // 4)
        lunch_selected = select_diverse(lunch_pool, max_recipes // 3)
        dinner_selected = select_diverse(dinner_pool, max_recipes // 3)
        
        # Fill remaining with any high-scoring recipes
        selected = breakfast_selected + lunch_selected + dinner_selected
        used_names = {r.name for r in selected}
        
        remaining_slots = max_recipes - len(selected)
        for item in scored:
            if remaining_slots <= 0:
                break
            if item['recipe'].name not in used_names:
                selected.append(item['recipe'])
                used_names.add(item['recipe'].name)
                remaining_slots -= 1
        
        # Shuffle to add variety between runs
        random.shuffle(selected)
        
        return selected[:max_recipes]
    
    def _parse_ai_response(self, response_text: str) -> list[dict]:
        """Parse AI response to extract meal plan."""
        # Try to find JSON in the response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fall back to parsing structured text
        meals = []
        current_day = 0
        
        for line in response_text.split("\n"):
            line = line.strip()
            
            # Check for day header
            day_match = re.search(r"Day\s*(\d+)", line, re.IGNORECASE)
            if day_match:
                current_day = int(day_match.group(1))
                continue
            
            # Check for meal entry
            meal_match = re.search(
                r"(breakfast|lunch|dinner|snack):\s*(.+)",
                line, re.IGNORECASE
            )
            if meal_match and current_day > 0:
                meal_type = meal_match.group(1).lower()
                recipe_name = meal_match.group(2).strip()
                
                # Try to find matching recipe
                matched_recipe = None
                for name in self.recipe_library.recipes:
                    if recipe_name.lower() in name.lower() or name.lower() in recipe_name.lower():
                        matched_recipe = name
                        break
                
                if matched_recipe:
                    meals.append({
                        "day": current_day,
                        "meal_type": meal_type,
                        "recipe": matched_recipe,
                        "servings": 1,
                    })
        
        return meals
    
    async def generate_meal_plan_async(
        self,
        days: int = 7,
        preferences: Optional[MealPlanConfig] = None,
    ) -> MealPlan:
        """Generate a meal plan using AI (async version)."""
        if not self.model:
            return self._generate_rule_based_plan(days, preferences)
        
        prefs = preferences or self.config.meal_plan_config
        
        # Filter recipes based on preferences
        filtered_recipes = self._filter_recipes(prefs)
        
        if not filtered_recipes:
            print("Warning: No recipes match the current filters.")
            filtered_recipes = list(self.recipe_library.recipes.values())[:50]
        
        # Build the prompt
        prompt = self._build_planning_prompt(days, prefs, filtered_recipes)
        
        try:
            response = await self.model.generate_content_async(prompt)
            planned_meals = self._parse_ai_response(response.text)
            meal_plan = self._build_meal_plan(planned_meals, days)
            return self._auto_scale_servings(meal_plan)
        except Exception as e:
            print(f"AI generation failed: {e}")
            plan = self._generate_rule_based_plan(days, prefs)
            return self._auto_scale_servings(plan)
    
    def generate_meal_plan(
        self,
        days: int = 7,
        preferences: Optional[MealPlanConfig] = None,
    ) -> MealPlan:
        """Generate a meal plan using AI (sync version)."""
        if not self.model:
            return self._generate_rule_based_plan(days, preferences)
        
        prefs = preferences or self.config.meal_plan_config
        
        # Filter recipes based on preferences
        filtered_recipes = self._filter_recipes(prefs)
        
        if not filtered_recipes:
            print("Warning: No recipes match the current filters.")
            filtered_recipes = list(self.recipe_library.recipes.values())[:50]
        
        # Build the prompt
        prompt = self._build_planning_prompt(days, prefs, filtered_recipes)
        
        try:
            response = self.model.generate_content(prompt)
            planned_meals = self._parse_ai_response(response.text)
            meal_plan = self._build_meal_plan(planned_meals, days)
            return self._auto_scale_servings(meal_plan)
        except Exception as e:
            print(f"AI generation failed: {e}")
            plan = self._generate_rule_based_plan(days, prefs)
            return self._auto_scale_servings(plan)
    
    def _filter_recipes(self, prefs: MealPlanConfig) -> list[Recipe]:
        """Filter recipes based on preferences."""
        recipes = list(self.recipe_library.recipes.values())
        
        # Filter by time
        if prefs.max_cook_time_minutes:
            max_time = timedelta(minutes=prefs.max_cook_time_minutes)
            recipes = [
                r for r in recipes
                if r.total_time is None or r.total_time <= max_time
            ]
        
        # Filter by mood preferences
        if prefs.mood_preferences:
            recipes = [r for r in recipes if r.matches_mood(prefs.mood_preferences)]
        
        # Filter by excluded ingredients
        if prefs.excluded_ingredients:
            recipes = [
                r for r in recipes
                if not any(
                    exc.lower() in ing_name
                    for exc in prefs.excluded_ingredients
                    for ing_name in r.get_ingredient_names()
                )
            ]
        
        return recipes
    
    def _build_planning_prompt(
        self,
        days: int,
        prefs: MealPlanConfig,
        recipes: list[Recipe],
    ) -> str:
        """Build the AI planning prompt."""
        targets = self.config.macro_targets
        
        mood_str = ""
        if prefs.mood_preferences:
            mood_str = f"\nPreferred flavors/moods: {', '.join(prefs.mood_preferences)}"
        
        cuisine_str = ""
        if prefs.cuisine_preferences:
            cuisine_str = f"\nPreferred cuisines: {', '.join(prefs.cuisine_preferences)}"
        
        pantry_context = self._build_pantry_context()
        
        # Smart pre-filtering: score recipes by pantry coverage + protein + macro fit
        scored_recipes = self._score_recipes_for_selection(recipes)
        recipe_context = self._build_recipe_context(scored_recipes)
        
        prompt = f"""You are a meal planning assistant. Create a {days}-day meal plan using the available recipes.

DAILY NUTRITIONAL TARGETS:
- Calories: {targets.calories:.0f}
- Protein: {targets.protein:.1f}g (prioritize hitting this target)
- Carbs: {targets.carbs:.1f}g
- Fat: {targets.fat:.1f}g

MEAL STRUCTURE:
- 3 main meals: breakfast, lunch, dinner (use ONE recipe from the list per meal - do NOT split into multiple items)
- 1-2 snacks: Use pantry items or suggest high-calorie nutritious snacks/smoothies

SNACK GUIDELINES:
- Target 300-600 calories per snack to help hit daily targets
- STRONGLY prefer snacks using pantry ingredients (milk, oats, peanut butter, eggs, yogurt, nuts, etc.)
- Mix healthy snacks with INDULGENT treats across the week:
  * Healthy: protein shakes, smoothies, yogurt parfaits, peanut butter toast, nuts, oatmeal
  * Indulgent (include at least 1-2 per week): ice cream, cookies, brownies, chips, candy bars, Pop-Tarts, chocolate, Oreos, donuts, pizza rolls
- For snacks NOT in the recipe list, include an "ingredients" array with specific amounts

CONSTRAINTS:
- Maximum prep/cook time per meal: {prefs.max_cook_time_minutes} minutes
- Prioritize variety - don't repeat recipes within 3 days
- Keep main meal servings reasonable (1-2x) since snacks help hit targets{mood_str}{cuisine_str}

{pantry_context}

AVAILABLE RECIPES:
{recipe_context}

PRIORITIES (in order of importance):
1. **MINIMIZE SHOPPING** - STRONGLY prefer recipes that use pantry ingredients (this is critical!)
2. Hit daily calorie and protein targets using meals + snacks
3. Keep main meal portions reasonable (1-1.5x servings)
4. Use snacks to fill calorie gaps (300-600 cal each)
5. Provide variety across the week

Respond with a JSON array in this format:
[
  {{"day": 1, "meal_type": "breakfast", "recipe": "Exact Recipe Name", "servings": 1.5}},
  {{"day": 1, "meal_type": "lunch", "recipe": "Exact Recipe Name", "servings": 1.5}},
  {{"day": 1, "meal_type": "dinner", "recipe": "Exact Recipe Name", "servings": 1.5}},
  {{"day": 1, "meal_type": "snack", "recipe": "Protein shake with banana, peanut butter, and oats", "servings": 1, "calories": 500, "protein": 30, "ingredients": ["2 scoops whey protein", "1 banana", "2 tbsp peanut butter", "1/2 cup oats", "1 cup milk"]}},
  {{"day": 1, "meal_type": "snack", "recipe": "Ice cream sundae", "servings": 1, "calories": 450, "protein": 8, "ingredients": ["2 scoops vanilla ice cream", "2 tbsp chocolate sauce", "whipped cream"]}},
  ...
]

For SNACKS: Include "calories", "protein", and "ingredients" (array of strings with amounts) since they may not be in the recipe list.
Use EXACT recipe names from the list for main meals. Plan all {days} days."""

        return prompt
    
    def _build_meal_plan(self, planned_meals: list[dict], days: int) -> MealPlan:
        """Build a MealPlan object from parsed AI response."""
        day_plans = {i: DayPlan(day=i, meals=[]) for i in range(1, days + 1)}
        
        for meal_data in planned_meals:
            day_num = meal_data.get("day", 1)
            if day_num not in day_plans:
                continue
            
            recipe_name = meal_data.get("recipe", "")
            recipe = self.recipe_library.get_recipe(recipe_name)
            
            if not recipe:
                # Try fuzzy match
                for name, r in self.recipe_library.recipes.items():
                    if recipe_name.lower() in name.lower():
                        recipe = r
                        break
            
            meal_type_str = meal_data.get("meal_type", "dinner").lower()
            meal_type = {
                "breakfast": MealType.BREAKFAST,
                "lunch": MealType.LUNCH,
                "dinner": MealType.DINNER,
                "snack": MealType.SNACK,
            }.get(meal_type_str, MealType.DINNER)
            
            if recipe:
                day_plans[day_num].meals.append(MealSlot(
                    day=day_num,
                    meal_type=meal_type,
                    recipe=recipe,
                    servings=float(meal_data.get("servings", 1)),
                ))
            elif meal_type == MealType.SNACK and recipe_name:
                # For snacks, create a simple recipe from AI suggestion
                snack_calories = float(meal_data.get("calories", 300))
                snack_protein = float(meal_data.get("protein", 15))
                
                # Parse ingredients from AI response
                snack_ingredients = []
                for ing_str in meal_data.get("ingredients", []):
                    snack_ingredients.append(Ingredient(name=ing_str))
                
                snack_recipe = Recipe(
                    name=recipe_name,
                    source_path=Path("snacks") / "ai-suggested.md",
                    ingredients=snack_ingredients,
                    instructions=[],
                    macros=Macros(
                        calories=snack_calories,
                        protein=snack_protein,
                        carbs=meal_data.get("carbs", 30),
                        fat=meal_data.get("fat", 15),
                    ),
                    meal_types=[MealType.SNACK],
                    is_ai_suggested=True,
                )
                
                day_plans[day_num].meals.append(MealSlot(
                    day=day_num,
                    meal_type=meal_type,
                    recipe=snack_recipe,
                    servings=float(meal_data.get("servings", 1)),
                ))
        
        return MealPlan(days=list(day_plans.values()))
    
    def _auto_scale_servings(self, meal_plan: MealPlan) -> MealPlan:
        """Auto-scale servings for each day to hit calorie/protein targets."""
        targets = self.config.macro_targets
        
        for day in meal_plan.days:
            if not day.meals:
                continue
            
            # Calculate current day's macros at 1x servings (only from real recipes, not AI snacks)
            base_macros = Macros()
            for meal in day.meals:
                if meal.recipe and not meal.recipe.is_ai_suggested:
                    base_macros = base_macros + meal.recipe.macros
            
            if base_macros.calories <= 0:
                continue
            
            # Account for AI-suggested snacks (they have fixed macros)
            snack_macros = Macros()
            for meal in day.meals:
                if meal.recipe and meal.recipe.is_ai_suggested:
                    snack_macros = snack_macros + (meal.recipe.macros * meal.servings)
            
            # Calculate how much we need from main meals after snacks
            remaining_calories = targets.calories - snack_macros.calories
            remaining_protein = targets.protein - snack_macros.protein
            
            if remaining_calories <= 0 or remaining_protein <= 0:
                # Snacks already hit targets, no scaling needed
                continue
            
            # Calculate scaling factor for main meals
            cal_scale = remaining_calories / base_macros.calories if base_macros.calories > 0 else 1
            protein_scale = remaining_protein / base_macros.protein if base_macros.protein > 0 else 1
            
            # Use the higher scale to ensure we hit protein, but cap at 3x to be realistic
            scale = min(max(cal_scale, protein_scale), 3.0)
            
            # Round to nice numbers (0.5 increments)
            scale = round(scale * 2) / 2
            scale = max(1.0, scale)  # At least 1x
            
            # Apply scaling only to non-AI meals
            for meal in day.meals:
                if meal.recipe and not meal.recipe.is_ai_suggested:
                    meal.servings = scale
        
        return meal_plan

    def _generate_rule_based_plan(
        self,
        days: int,
        preferences: Optional[MealPlanConfig] = None,
    ) -> MealPlan:
        """Generate a meal plan using rule-based logic (fallback)."""
        prefs = preferences or self.config.meal_plan_config
        targets = self.config.macro_targets
        
        # Calculate per-meal targets (roughly split across 3 meals)
        # Breakfast: ~25%, Lunch: ~35%, Dinner: ~40%
        meal_cal_limits = {
            MealType.BREAKFAST: targets.calories * 0.35,  # Allow some flexibility
            MealType.LUNCH: targets.calories * 0.45,
            MealType.DINNER: targets.calories * 0.50,
        }
        
        # Categorize recipes by meal type
        breakfast_recipes = self.recipe_library.search(meal_types=[MealType.BREAKFAST])
        lunch_recipes = self.recipe_library.search(meal_types=[MealType.LUNCH])
        dinner_recipes = self.recipe_library.search(meal_types=[MealType.DINNER])
        
        # Filter out recipes with extreme calories (> 1000 cal) for individual meals
        max_single_meal_cal = targets.calories * 0.6  # Max 60% of daily in one meal
        breakfast_recipes = [r for r in breakfast_recipes if r.macros.calories <= meal_cal_limits[MealType.BREAKFAST]]
        lunch_recipes = [r for r in lunch_recipes if r.macros.calories <= meal_cal_limits[MealType.LUNCH]]
        dinner_recipes = [r for r in dinner_recipes if r.macros.calories <= meal_cal_limits[MealType.DINNER]]
        
        # Filter by time preferences
        if prefs.max_cook_time_minutes:
            max_time = timedelta(minutes=prefs.max_cook_time_minutes)
            breakfast_recipes = [r for r in breakfast_recipes if r.total_time is None or r.total_time <= max_time]
            lunch_recipes = [r for r in lunch_recipes if r.total_time is None or r.total_time <= max_time]
            dinner_recipes = [r for r in dinner_recipes if r.total_time is None or r.total_time <= max_time]
        
        # Filter by preferences
        if prefs.mood_preferences:
            breakfast_recipes = [r for r in breakfast_recipes if r.matches_mood(prefs.mood_preferences)]
            lunch_recipes = [r for r in lunch_recipes if r.matches_mood(prefs.mood_preferences)]
            dinner_recipes = [r for r in dinner_recipes if r.matches_mood(prefs.mood_preferences)]
        
        # Sort by protein-to-calorie ratio (efficiency)
        def protein_efficiency(r):
            if r.macros.calories == 0:
                return 0
            return r.macros.protein / r.macros.calories
        
        breakfast_recipes.sort(key=protein_efficiency, reverse=True)
        lunch_recipes.sort(key=protein_efficiency, reverse=True)
        dinner_recipes.sort(key=protein_efficiency, reverse=True)
        
        day_plans = []
        used_recipes = set()
        
        for day_num in range(1, days + 1):
            meals = []
            current_macros = Macros()
            
            # Pick breakfast
            breakfast = self._pick_best_recipe(
                breakfast_recipes, current_macros, targets, used_recipes
            )
            if breakfast:
                meals.append(MealSlot(
                    day=day_num, meal_type=MealType.BREAKFAST,
                    recipe=breakfast, servings=1.0
                ))
                current_macros = current_macros + breakfast.macros
                used_recipes.add(breakfast.name)
            
            # Pick lunch
            lunch = self._pick_best_recipe(
                lunch_recipes, current_macros, targets, used_recipes
            )
            if lunch:
                meals.append(MealSlot(
                    day=day_num, meal_type=MealType.LUNCH,
                    recipe=lunch, servings=1.0
                ))
                current_macros = current_macros + lunch.macros
                used_recipes.add(lunch.name)
            
            # Pick dinner
            dinner = self._pick_best_recipe(
                dinner_recipes, current_macros, targets, used_recipes
            )
            if dinner:
                meals.append(MealSlot(
                    day=day_num, meal_type=MealType.DINNER,
                    recipe=dinner, servings=1.0
                ))
                used_recipes.add(dinner.name)
            
            day_plans.append(DayPlan(day=day_num, meals=meals))
            
            # Allow recipe reuse after 3 days
            if day_num >= 3:
                old_day = day_plans[day_num - 3]
                for meal in old_day.meals:
                    if meal.recipe:
                        used_recipes.discard(meal.recipe.name)
        
        return MealPlan(days=day_plans)
    
    def _calculate_pantry_coverage(self, recipe: Recipe) -> float:
        """Calculate what percentage of recipe ingredients are in pantry (0.0 to 1.0)."""
        if not self.pantry or not recipe.ingredients:
            return 0.0
        
        total_ingredients = len(recipe.ingredients)
        in_pantry = 0
        
        for ingredient in recipe.ingredients:
            if self.pantry.has_item(ingredient.name):
                in_pantry += 1
        
        return in_pantry / total_ingredients if total_ingredients > 0 else 0.0
    
    def _pick_best_recipe(
        self,
        candidates: list[Recipe],
        current_macros: Macros,
        targets: MacroTargets,
        used_recipes: set,
    ) -> Optional[Recipe]:
        """Pick the best recipe based on macro fit AND pantry coverage."""
        available = [r for r in candidates if r.name not in used_recipes]
        
        if not available:
            # Fall back to all candidates if we've used everything
            available = candidates
        
        if not available:
            return None
        
        # Score each recipe: combine macro fit (60%) + pantry coverage (40%)
        pantry_weight = self.config.meal_plan_config.pantry_weight  # Default 0.8
        
        scored = []
        for r in available:
            macro_score = self.macro_tracker.score_recipe_fit(r, current_macros)
            pantry_score = self._calculate_pantry_coverage(r) * 100  # Scale to similar range
            
            # Combined score: macro fit matters, but pantry coverage gives big bonus
            # Higher pantry_weight = more bias towards using pantry items
            combined_score = macro_score * (1 - pantry_weight * 0.5) + pantry_score * pantry_weight
            scored.append((r, combined_score, macro_score, pantry_score))
        
        # Sort by combined score and pick top
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None
    
    def suggest_snacks(
        self,
        current_macros: Macros,
        count: int = 5,
    ) -> list[dict]:
        """Suggest snacks to fill macro gaps."""
        targets = self.config.macro_targets
        remaining_protein = targets.protein - current_macros.protein
        remaining_calories = targets.calories - current_macros.calories
        
        # Built-in snack suggestions (not from recipe library)
        snack_options = [
            {"name": "Greek Yogurt (1 cup)", "calories": 130, "protein": 17, "carbs": 9, "fat": 0},
            {"name": "Protein Shake", "calories": 150, "protein": 25, "carbs": 5, "fat": 2},
            {"name": "Hard Boiled Eggs (2)", "calories": 140, "protein": 12, "carbs": 1, "fat": 10},
            {"name": "Cottage Cheese (1 cup)", "calories": 180, "protein": 24, "carbs": 8, "fat": 5},
            {"name": "Almonds (1 oz)", "calories": 165, "protein": 6, "carbs": 6, "fat": 14},
            {"name": "String Cheese (2)", "calories": 160, "protein": 14, "carbs": 2, "fat": 12},
            {"name": "Turkey Slices (4 oz)", "calories": 120, "protein": 24, "carbs": 0, "fat": 2},
            {"name": "Hummus + Veggies", "calories": 150, "protein": 5, "carbs": 15, "fat": 8},
            {"name": "Peanut Butter Toast", "calories": 250, "protein": 10, "carbs": 25, "fat": 14},
            {"name": "Beef Jerky (1 oz)", "calories": 80, "protein": 13, "carbs": 3, "fat": 1},
            {"name": "Edamame (1 cup)", "calories": 190, "protein": 17, "carbs": 15, "fat": 8},
            {"name": "Protein Bar", "calories": 200, "protein": 20, "carbs": 20, "fat": 8},
        ]
        
        # Score snacks based on what's needed
        for snack in snack_options:
            score = 0
            
            # Reward high protein if needed
            if remaining_protein > 0:
                score += (snack["protein"] / remaining_protein) * 50
            
            # Reward staying within calorie budget
            if remaining_calories > 0 and snack["calories"] <= remaining_calories:
                score += 30
            elif snack["calories"] > remaining_calories:
                score -= 20
            
            snack["score"] = score
            snack["fills_protein"] = f"{snack['protein']/remaining_protein*100:.0f}%" if remaining_protein > 0 else "N/A"
        
        # Sort by score and return top N
        snack_options.sort(key=lambda x: x["score"], reverse=True)
        return snack_options[:count]
    
    def generate_shopping_list(self, meal_plan: MealPlan) -> ShoppingList:
        """Generate a shopping list from a meal plan."""
        return self.shopping_generator.generate(meal_plan)
    
    def get_plan_summary(self, meal_plan: MealPlan) -> str:
        """Generate a summary of the meal plan."""
        lines = ["# 📅 Meal Plan Summary\n"]
        
        for day in meal_plan.days:
            day_macros = day.get_total_macros()
            analysis = self.macro_tracker.analyze_day(day)
            
            lines.append(f"## Day {day.day}")
            lines.append(f"*{day_macros.calories:.0f} cal | {day_macros.protein:.1f}g protein*\n")
            
            # Custom order: breakfast -> lunch -> dinner -> snack -> dessert
            meal_order = {
                MealType.BREAKFAST: 0,
                MealType.LUNCH: 1,
                MealType.DINNER: 2,
                MealType.SNACK: 3,
                MealType.DESSERT: 4,
            }
            for meal in sorted(day.meals, key=lambda m: meal_order.get(m.meal_type, 5)):
                if meal.recipe:
                    emoji = {
                        MealType.BREAKFAST: "🌅",
                        MealType.LUNCH: "☀️",
                        MealType.DINNER: "🌙",
                        MealType.SNACK: "🍎",
                        MealType.DESSERT: "🍰",
                    }.get(meal.meal_type, "🍽️")
                    
                    time_str = ""
                    if meal.recipe.total_time:
                        mins = int(meal.recipe.total_time.total_seconds() / 60)
                        time_str = f" ({mins} min)"
                    
                    # Show servings if not 1x (format nicely)
                    if meal.servings != 1.0:
                        servings_str = f" [{meal.servings:.1f}x]" if meal.servings != int(meal.servings) else f" [{int(meal.servings)}x]"
                    else:
                        servings_str = ""
                    
                    # For AI-suggested snacks, don't make a link
                    if meal.recipe.is_ai_suggested:
                        recipe_display = meal.recipe.name
                    else:
                        # Create link to recipe file (relative path for Obsidian)
                        recipe_display = f"[[{meal.recipe.source_path.stem}]]"
                    
                    lines.append(
                        f"- {emoji} **{meal.meal_type.value.title()}**: {recipe_display}{time_str}{servings_str}"
                    )
                    
                    # Show scaled macros
                    scaled_cals = meal.recipe.macros.calories * meal.servings
                    scaled_protein = meal.recipe.macros.protein * meal.servings
                    lines.append(
                        f"  - {scaled_cals:.0f} cal, {scaled_protein:.1f}g protein"
                    )
                    
                    # For AI-suggested snacks, show ingredients if available
                    if meal.recipe.is_ai_suggested and meal.recipe.ingredients:
                        for ing in meal.recipe.ingredients:
                            lines.append(f"    - {ing.name}")
            
            # Show macro status
            status = "✅" if analysis.is_within_target() else "⚠️"
            lines.append(f"\n{status} Protein: {analysis.protein_pct:.0f}% of target\n")
        
        # Overall summary
        avg_analysis = self.macro_tracker.get_average_analysis(meal_plan)
        lines.append("\n---")
        lines.append("## 📊 Weekly Average")
        lines.append(avg_analysis.to_summary())
        
        return "\n".join(lines)
    
    def optimize_for_meal_prep(self, meal_plan: MealPlan) -> dict:
        """Analyze meal plan for meal prep opportunities."""
        prep_recipes = []
        prep_suggestions = []
        
        # Find recipes that can be prepped
        for day in meal_plan.days:
            for meal in day.meals:
                if meal.recipe and meal.recipe.can_meal_prep:
                    prep_recipes.append({
                        "recipe": meal.recipe.name,
                        "day": day.day,
                        "prep_days": meal.recipe.prep_ahead_days,
                    })
        
        # Group by recipe for batch cooking suggestions
        recipe_counts = {}
        for pr in prep_recipes:
            name = pr["recipe"]
            if name not in recipe_counts:
                recipe_counts[name] = {"count": 0, "days": []}
            recipe_counts[name]["count"] += 1
            recipe_counts[name]["days"].append(pr["day"])
        
        for recipe_name, info in recipe_counts.items():
            if info["count"] >= 2:
                prep_suggestions.append({
                    "recipe": recipe_name,
                    "suggestion": f"Batch cook for days {info['days']}",
                    "time_saved": "~30 min",
                })
        
        return {
            "prep_recipes": prep_recipes,
            "batch_suggestions": prep_suggestions,
            "total_preppable": len(prep_recipes),
        }
