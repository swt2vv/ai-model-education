#%%
# ============================================================
# MODEL C — TEACHING ACTION + EXPLANATION GENERATOR
# ============================================================

def choose_teaching_action(learning_style, tone):
    """
    learning_style: int from Model B (0–4)
    tone: int from Model A (0–5)
    """

    # Base action from learning style
    base_actions = {
        0: "visual_explanation",
        1: "verbal_explanation",
        2: "give_example",
        3: "step_by_step",
        4: "ask_question"
    }

    action = base_actions.get(learning_style, "verbal_explanation")

    # Tone adjustments
    if tone == 5:  # confused
        action = "step_by_step"
    elif tone == 4:  # frustrated
        action = "simplify"
    elif tone == 3:  # bored
        action = "increase_engagement"
    elif tone == 2:  # excited
        action = "increase_difficulty"

    return action


def generate_explanation(action, concept="adding fractions"):
    """
    Converts the chosen action into a natural-language explanation.
    """

    templates = {
        "visual_explanation": f"Imagine {concept} like pieces of a picture. Let’s visualize it together.",
        "verbal_explanation": f"Let’s talk through {concept} clearly and simply.",
        "give_example": f"Here’s an example to help with {concept}: If you have 1/2 and add 1/4...",
        "step_by_step": f"Let’s break {concept} into small steps so it feels easier.",
        "ask_question": f"What part of {concept} feels tricky to you? Let’s explore it together.",
        "simplify": f"Let’s slow down and make {concept} easier. Here’s the simplest way to see it...",
        "increase_engagement": f"Let’s make {concept} more interesting — try this challenge!",
        "increase_difficulty": f"You’re doing great. Let’s try a slightly harder version of {concept}."
    }

    return templates.get(action, f"Let’s explore {concept} together.")


def model_c_generate(learning_style, tone, concept="adding fractions"):
    """
    Full Model C pipeline:
    - choose action
    - generate explanation
    """

    action = choose_teaching_action(learning_style, tone)
    explanation = generate_explanation(action, concept)

    return {
        "action": action,
        "explanation": explanation
    }


result = model_c_generate(
    learning_style=1,   # example-driven
    tone=2,             # excited
    concept="adding fractions"
)

print(result)

#%%