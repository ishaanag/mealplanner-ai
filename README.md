# MealPlanner AI 🍽️

An AI-powered meal planning tool that creates personalized weekly meal plans based on your recipes, nutritional targets, pantry inventory, and preferences.

## Features

- **📚 Recipe Parsing**: Reads recipes from markdown files with YAML frontmatter (Obsidian-compatible)
- **🎯 Macro Tracking**: Hit your daily protein, calorie, carbs, and fat targets
- **🗄️ Pantry Management**: Track what you have and prioritize using those ingredients
- **🛒 Shopping Lists**: Auto-generated shopping lists with pantry deduction
- **🤖 AI-Powered Planning**: Uses Google Gemini for intelligent meal selection
- **😋 Mood-Based Filtering**: Filter by flavor preferences (spicy, savory, sweet, etc.)
- **⏱️ Time-Aware**: Respects cooking time constraints
- **🍎 Snack Suggestions**: Fill macro gaps with smart snack recommendations
- **🍳 Meal Prep Optimization**: Batch cooking suggestions with prep schedules to save time

## Installation

```bash
cd mealplanner-ai
pip install -e .
```

## Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your settings:**
   ```
   GEMINI_API_KEY=your_api_key_here
   RECIPES_PATH=/path/to/your/recipes
   PANTRY_PATH=/path/to/your/pantry.md
   ```

3. **Initialize the tool:**
   ```bash
   mealplan init
   ```

## Recipe Format

Your recipes should be markdown files with YAML frontmatter:

```markdown
---
tags: [recipe, dinner]
source: https://example.com/recipe
time: PT30M
yields: 4 servings
calories: 450
protein: 35
carbs: 25
fat: 20
---

# Recipe Name

## Ingredients
- [ ] Chicken breast, 500 g
- [ ] Olive oil, 2 tablespoons
- [ ] Garlic, 3 cloves

## Instructions
1. First step...
2. Second step...
```

## Usage

### Generate a Meal Plan

```bash
# Basic 7-day plan
mealplan plan

# Custom targets
mealplan plan --days 5 --calories 2200 --protein 180

# With mood preferences
mealplan plan --mood spicy --mood savory

# Save to file
mealplan plan -o meal-plan.md
```

### Search Recipes

```bash
# Search by name
mealplan search chicken

# Filter by meal type
mealplan search --meal-type breakfast

# Filter by mood and protein
mealplan search --mood spicy --min-protein 30
```

### Manage Pantry

```bash
# View pantry
mealplan pantry

# Add items
mealplan add-pantry "Chicken breast" -q 2 -u lbs
```

### Get Snack Suggestions

```bash
# Based on remaining macros
mealplan snacks -d 1200 75 150 40  # cal protein carbs fat
```

## Python API

```python
from mealplanner import (
    Config, RecipeLibrary, Pantry, MealPlanner,
    MacroTargets, MealPlanConfig
)

# Load configuration
config = Config.from_env()
config.macro_targets = MacroTargets(
    calories=2000,
    protein=150,
    carbs=200,
    fat=65
)

# Load recipes and pantry
recipes = RecipeLibrary(config.recipes_path)
pantry = Pantry(config.pantry_path)

# Create planner
planner = MealPlanner(config, recipes, pantry)

# Generate a meal plan
plan = planner.generate_meal_plan(
    days=7,
    preferences=MealPlanConfig(
        mood_preferences=["spicy", "savory"],
        max_cook_time_minutes=45
    )
)

# Get summary and shopping list
print(planner.get_plan_summary(plan))
shopping = planner.generate_shopping_list(plan)
print(shopping.to_markdown())
```

## Pantry File Format

Create a markdown file for your pantry inventory:

```markdown
---
tags: [pantry, inventory]
updated: 2026-02-28
---

# Pantry Inventory

## Protein
- Chicken breast, 2 lbs
- Ground beef, 1 lb
- Eggs, 12

## Produce
- Onions, 3
- Garlic, 1 head
- Tomatoes, 4

## Spices
- Salt
- Black pepper
- Cumin
```

## Roadmap

- [ ] Calendar integration for scheduling cooking time
- [x] Meal prep optimization (batch cooking suggestions)
- [ ] Grocery store integration
- [ ] Recipe scaling
- [ ] Nutritional balance analysis over time
- [ ] Support for dietary restrictions

## License

MIT
