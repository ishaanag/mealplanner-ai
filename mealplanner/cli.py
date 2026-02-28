"""Command-line interface for MealPlanner AI."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from .config import Config, MacroTargets, MealPlanConfig
from .recipe_parser import RecipeLibrary
from .pantry import Pantry, create_sample_pantry
from .planner import MealPlanner
from .macro_tracker import MacroTracker
from .shopping import ShoppingListGenerator

console = Console()


def load_app(env_path: str = None):
    """Load application components."""
    config = Config.from_env(Path(env_path) if env_path else None)
    
    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[yellow]Warning:[/yellow] {error}")
    
    recipe_library = RecipeLibrary(config.recipes_path)
    
    pantry = None
    if config.pantry_path.exists():
        pantry = Pantry(config.pantry_path)
    
    planner = MealPlanner(config, recipe_library, pantry)
    
    return config, recipe_library, pantry, planner


@click.group()
@click.option("--env", default=None, help="Path to .env file")
@click.pass_context
def cli(ctx, env):
    """🍽️  MealPlanner AI - Smart meal planning with macro tracking."""
    ctx.ensure_object(dict)
    ctx.obj["env"] = env


@cli.command()
@click.option("--days", default=7, help="Number of days to plan")
@click.option("--calories", default=2000, help="Daily calorie target")
@click.option("--protein", default=150, help="Daily protein target (g)")
@click.option("--carbs", default=200, help="Daily carbs target (g)")
@click.option("--fat", default=65, help="Daily fat target (g)")
@click.option("--mood", multiple=True, help="Flavor preferences (spicy, savory, sweet, etc.)")
@click.option("--max-time", default=60, help="Maximum cooking time in minutes")
@click.option("--servings", default=1.0, help="Serving multiplier for all meals (e.g., 1.5 for larger portions)")
@click.option("--output", "-o", default=None, help="Output file for meal plan")
@click.option("--shopping", "-s", default=None, help="Output file for shopping list (separate .md)")
@click.pass_context
def plan(ctx, days, calories, protein, carbs, fat, mood, max_time, servings, output, shopping):
    """Generate a meal plan based on your preferences."""
    config, recipe_library, pantry, planner = load_app(ctx.obj["env"])
    
    # Update config with CLI options
    config.macro_targets = MacroTargets(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
    )
    
    config.meal_plan_config = MealPlanConfig(
        days=days,
        max_cook_time_minutes=max_time,
        mood_preferences=list(mood),
    )
    
    # Recreate planner with updated config
    planner = MealPlanner(config, recipe_library, pantry)
    
    console.print(f"\n[bold blue]🍽️  Generating {days}-day meal plan...[/bold blue]\n")
    console.print(f"Targets: {calories} cal | {protein}g protein | {carbs}g carbs | {fat}g fat")
    
    if mood:
        console.print(f"Mood preferences: {', '.join(mood)}")
    
    if servings != 1.0:
        console.print(f"Serving multiplier: {servings}x")
    
    console.print()
    
    # Generate plan
    meal_plan = planner.generate_meal_plan(days=days)
    
    # Apply serving multiplier if specified
    if servings != 1.0:
        for day in meal_plan.days:
            for meal in day.meals:
                meal.servings *= servings
    
    # Display summary
    summary = planner.get_plan_summary(meal_plan)
    console.print(Markdown(summary))
    
    # Check if we're significantly under targets and suggest snacks
    from .models import Macros
    total_macros = Macros()
    for day in meal_plan.days:
        total_macros = total_macros + day.get_total_macros()
    avg_protein = total_macros.protein / days if days > 0 else 0
    avg_calories = total_macros.calories / days if days > 0 else 0
    
    protein_pct = (avg_protein / protein) * 100 if protein > 0 else 100
    calorie_pct = (avg_calories / calories) * 100 if calories > 0 else 100
    
    if protein_pct < 80 or calorie_pct < 80:
        console.print("\n[yellow]⚠️  Plan is under targets. Suggested additions:[/yellow]")
        remaining_protein = protein - avg_protein
        remaining_cals = calories - avg_calories
        
        snacks = planner.suggest_snacks(Macros(calories=avg_calories, protein=avg_protein), count=5)
        console.print(f"\n[bold]Add daily to hit targets (need ~{remaining_protein:.0f}g more protein, ~{remaining_cals:.0f} more cal):[/bold]")
        for snack in snacks[:3]:
            console.print(f"  • {snack['name']}: {snack['protein']}g protein, {snack['calories']} cal")
        
        console.print(f"\n[dim]Tip: Use --servings 1.5 or --servings 2 for larger portions[/dim]")
    
    # Generate shopping list
    shopping_list = planner.generate_shopping_list(meal_plan)
    shopping_md = planner.shopping_generator.to_markdown(shopping_list)
    
    console.print("\n")
    console.print(Markdown(shopping_md))
    
    # Save meal plan to file if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            f.write(summary)
            if not shopping:  # Include shopping list in meal plan if not saving separately
                f.write("\n\n---\n\n")
                f.write(shopping_md)
        console.print(f"\n[green]✅ Saved meal plan to {output}[/green]")
    
    # Save shopping list to separate file if requested
    if shopping:
        shopping_path = Path(shopping)
        clean_shopping_md = planner.shopping_generator.to_clean_markdown(shopping_list)
        with open(shopping_path, "w") as f:
            f.write(clean_shopping_md)
        console.print(f"[green]✅ Saved shopping list to {shopping}[/green]")


@cli.command()
@click.pass_context
def recipes(ctx):
    """List all available recipes with stats."""
    config, recipe_library, _, _ = load_app(ctx.obj["env"])
    
    stats = recipe_library.get_stats()
    
    console.print(Panel(
        f"[bold]Recipe Library Stats[/bold]\n\n"
        f"Total recipes: {stats['total']}\n"
        f"With macro data: {stats['with_macros']}\n"
        f"Avg protein: {stats['avg_protein']:.1f}g\n"
        f"Avg calories: {stats['avg_calories']:.0f}\n"
        f"Meal prep suitable: {stats['meal_prep_suitable']}",
        title="📚 Recipes"
    ))
    
    # Show high protein recipes
    high_protein = recipe_library.get_high_protein_recipes(min_protein=30)[:10]
    
    table = Table(title="🏋️ High Protein Recipes (30g+)")
    table.add_column("Recipe", style="cyan")
    table.add_column("Protein", justify="right", style="green")
    table.add_column("Calories", justify="right")
    table.add_column("Time", justify="right")
    
    for recipe in sorted(high_protein, key=lambda r: r.macros.protein, reverse=True):
        time_str = ""
        if recipe.total_time:
            time_str = f"{int(recipe.total_time.total_seconds()/60)} min"
        table.add_row(
            recipe.name[:40],
            f"{recipe.macros.protein:.1f}g",
            f"{recipe.macros.calories:.0f}",
            time_str
        )
    
    console.print(table)


@cli.command()
@click.argument("query", required=False)
@click.option("--meal-type", type=click.Choice(["breakfast", "lunch", "dinner", "snack"]))
@click.option("--mood", multiple=True, help="Filter by mood (spicy, savory, sweet, etc.)")
@click.option("--max-time", type=int, help="Maximum cooking time in minutes")
@click.option("--min-protein", type=float, help="Minimum protein content")
@click.option("--max-calories", type=float, help="Maximum calories")
@click.pass_context
def search(ctx, query, meal_type, mood, max_time, min_protein, max_calories):
    """Search recipes with filters."""
    config, recipe_library, _, _ = load_app(ctx.obj["env"])
    
    from .models import MealType
    
    meal_types = None
    if meal_type:
        meal_types = [MealType(meal_type)]
    
    results = recipe_library.search(
        query=query,
        meal_types=meal_types,
        moods=list(mood) if mood else None,
        max_time_minutes=max_time,
        min_protein=min_protein,
        max_calories=max_calories,
    )
    
    if not results:
        console.print("[yellow]No recipes found matching your criteria.[/yellow]")
        return
    
    table = Table(title=f"🔍 Search Results ({len(results)} recipes)")
    table.add_column("Recipe", style="cyan")
    table.add_column("Protein", justify="right", style="green")
    table.add_column("Calories", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Flavors", style="dim")
    
    for recipe in results[:20]:
        time_str = ""
        if recipe.total_time:
            time_str = f"{int(recipe.total_time.total_seconds()/60)} min"
        
        flavors = ", ".join(p.value for p in recipe.flavor_profiles[:2])
        
        table.add_row(
            recipe.name[:40],
            f"{recipe.macros.protein:.1f}g",
            f"{recipe.macros.calories:.0f}",
            time_str,
            flavors
        )
    
    console.print(table)
    
    if len(results) > 20:
        console.print(f"\n[dim]...and {len(results) - 20} more results[/dim]")


@cli.command()
@click.pass_context
def pantry(ctx):
    """Show pantry inventory."""
    config, _, pantry, _ = load_app(ctx.obj["env"])
    
    if not pantry:
        console.print("[yellow]No pantry inventory found.[/yellow]")
        console.print(f"Create one at: {config.pantry_path}")
        
        if click.confirm("Would you like to create a sample pantry?"):
            create_sample_pantry(config.pantry_path)
            console.print(f"[green]✅ Created sample pantry at {config.pantry_path}[/green]")
        return
    
    stats = pantry.get_stats()
    
    console.print(Panel(
        f"[bold]Pantry Inventory[/bold]\n\n"
        f"Total items: {stats['total_items']}\n"
        f"Expiring soon: {stats['expiring_soon']}",
        title="🗄️ Pantry"
    ))
    
    # Show items by category
    for category, count in sorted(stats["by_category"].items()):
        items = pantry.get_by_category(category)
        item_names = [i.name for i in items[:5]]
        more = f" (+{count - 5} more)" if count > 5 else ""
        console.print(f"[bold]{category.title()}:[/bold] {', '.join(item_names)}{more}")


@cli.command()
@click.argument("item_name")
@click.option("--quantity", "-q", type=float, help="Quantity to add")
@click.option("--unit", "-u", help="Unit (lbs, oz, etc.)")
@click.pass_context
def add_pantry(ctx, item_name, quantity, unit):
    """Add an item to your pantry."""
    config, _, pantry, _ = load_app(ctx.obj["env"])
    
    if not pantry:
        if not config.pantry_path.parent.exists():
            config.pantry_path.parent.mkdir(parents=True)
        pantry = Pantry(config.pantry_path)
    
    item = pantry.add_item(item_name, quantity, unit)
    console.print(f"[green]✅ Added {item.name} to pantry[/green]")


@cli.command()
@click.option("--day-macros", "-d", nargs=4, type=float, help="Current day macros: cal protein carbs fat")
@click.option("--count", "-n", default=5, help="Number of suggestions")
@click.pass_context
def snacks(ctx, day_macros, count):
    """Suggest snacks to fill macro gaps."""
    config, recipe_library, pantry, planner = load_app(ctx.obj["env"])
    
    from .models import Macros
    
    if day_macros:
        current = Macros(
            calories=day_macros[0],
            protein=day_macros[1],
            carbs=day_macros[2],
            fat=day_macros[3],
        )
    else:
        # Assume halfway through the day
        current = Macros(
            calories=config.macro_targets.calories * 0.5,
            protein=config.macro_targets.protein * 0.5,
            carbs=config.macro_targets.carbs * 0.5,
            fat=config.macro_targets.fat * 0.5,
        )
    
    suggestions = planner.suggest_snacks(current, count)
    
    remaining_protein = config.macro_targets.protein - current.protein
    remaining_calories = config.macro_targets.calories - current.calories
    
    console.print(Panel(
        f"[bold]Remaining Daily Targets[/bold]\n\n"
        f"Protein: {remaining_protein:.1f}g\n"
        f"Calories: {remaining_calories:.0f}",
        title="🎯 Targets"
    ))
    
    table = Table(title="🍎 Suggested Snacks")
    table.add_column("Snack", style="cyan")
    table.add_column("Protein", justify="right", style="green")
    table.add_column("Calories", justify="right")
    table.add_column("Fills Protein", justify="right", style="yellow")
    
    for snack in suggestions:
        table.add_row(
            snack["name"],
            f"{snack['protein']}g",
            f"{snack['calories']}",
            snack["fills_protein"]
        )
    
    console.print(table)


@cli.command()
@click.pass_context  
def init(ctx):
    """Initialize MealPlanner AI with sample files."""
    config = Config.from_env()
    
    console.print("[bold blue]🚀 Initializing MealPlanner AI[/bold blue]\n")
    
    # Create .env file if it doesn't exist
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        env_example = Path(__file__).parent / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_path)
            console.print(f"[green]✅ Created .env file[/green]")
            console.print(f"   Please edit {env_path} and add your GEMINI_API_KEY")
    
    # Create sample pantry if needed
    if not config.pantry_path.exists():
        if click.confirm(f"Create sample pantry at {config.pantry_path}?", default=True):
            config.pantry_path.parent.mkdir(parents=True, exist_ok=True)
            create_sample_pantry(config.pantry_path)
            console.print(f"[green]✅ Created sample pantry[/green]")
    
    # Check recipe path
    if config.recipes_path.exists():
        recipe_count = len(list(config.recipes_path.rglob("*.md")))
        console.print(f"[green]✅ Found {recipe_count} recipes in {config.recipes_path}[/green]")
    else:
        console.print(f"[yellow]⚠️  Recipe path not found: {config.recipes_path}[/yellow]")
        console.print("   Update RECIPES_PATH in your .env file")
    
    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\nTry these commands:")
    console.print("  mealplan recipes      # See your recipe library stats")
    console.print("  mealplan plan         # Generate a meal plan")
    console.print("  mealplan search pizza # Search for recipes")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
