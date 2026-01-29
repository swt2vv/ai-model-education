from sklearn.tree import DecisionTreeClassifier
import pandas as pd

reward_history = []

#these maps convert our categorical features of the  user into numerical values for the model
#they are defined by integers in order for the model to read them
difficulty_map = {
    "Foundational": 0,
    "Basic": 1,
    "Standard": 2,
    "Advanced": 3
}

age_map = {
    "6": 0,
    "7": 1,
    "8": 2,
    "9": 3,
    "10": 4,
    "11": 5,
    "12": 6
}

style_map = {
    "visual": 0,
    "example": 1,
    "story": 2,
    "step": 3
}

confusion_phrases = [
    "can you explain again",
    "wait, what",
    "i don't get it",
    "i dont get it",
    "what do you mean",
    "i'm confused",
    "im confused"
]

#this is our sample data that we will use to train our initial model
data = {
    "incorrect_answer": [1, 1, 1, 1, 1, 1, 1, 1,  
                         0, 0, 0, 0,
                         0, 0, 0, 0 ],

    "correct_answer": [0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1,
                       0, 0, 0, 0],

    "slow_response": [1, 1, 0, 0, 0, 0, 0, 0,  
                      0, 1, 0, 0,
                      0, 0, 0, 0],

    "fast_response": [0, 0, 1, 1, 1, 1, 1, 1,
                      1, 0, 1, 1,
                      1, 1, 1, 1],

    "repeated_question": [0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 0],

    "negative_response": [1, 0, 0, 1, 0, 0, 0, 0, 
                          0, 0, 0, 0,
                          0, 0, 0, 0],

    "positive_response": [0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 0, 0,
                          0, 0, 0, 0],

    "difficulty_level": [
        difficulty_map["Foundational"],
        difficulty_map["Basic"],
        difficulty_map["Standard"],
        difficulty_map["Foundational"],
        difficulty_map["Standard"],
        difficulty_map["Advanced"], 
        difficulty_map["Foundational"],
        difficulty_map["Standard"],

        difficulty_map["Standard"],
        difficulty_map["Basic"],
        difficulty_map["Standard"],
        difficulty_map["Advanced"],

        difficulty_map["Foundational"],
        difficulty_map["Advanced"],
        difficulty_map["Basic"],
        difficulty_map["Standard"],
    ],

    "age_group": [
        age_map["7"],
        age_map["10"],
        age_map["11"],
        age_map["6"],
        age_map["8"],
        age_map["12"],
        age_map["9"],
        age_map["11"],

        age_map["10"],
        age_map["7"],
        age_map["9"],
        age_map["12"],

        age_map["6"],
        age_map["12"],
        age_map["8"],
        age_map["11"],
    ],

    "preferred_style": [
        style_map["visual"],   # 1 incorrect + slow + frustrated + 7 + visual learner
        style_map["example"],  # 2 incorrect + slow + neutral + 10 + example learner
        style_map["step"],     # 3 incorrect + fast + neutral + 11 + step learner
        style_map["visual"],   # 4 incorrect + fast + frustrated + 6 + visual learner
        style_map["example"],  # 5 incorrect + repeated question + 8 + example learner
        style_map["story"],    # 6 incorrect + advanced difficulty + 12 + story learner
        style_map["step"],     # 7 incorrect + step learner + 9 + step learner
        style_map["visual"],   # 8 incorrect + older child + 11 + visual learner

        style_map["example"],  # 9 correct + fast + happy + 10 + example learner
        style_map["visual"],   # 10 correct + slow + happy + 7 + visual learner
        style_map["step"],     # 11 correct + repeated question + 9 + step learner
        style_map["story"],    # 12 correct + neutral + 12 + story learner

        style_map["visual"],   # 13 neutral + foundational + 6 + visual learner
        style_map["story"],    # 14 neutral + advanced + 12 + story learner
        style_map["visual"],   # 15 neutral + visual learner + 8 + visual learner
        style_map["story"],],    # 16 neutral + story learner + 11 + story learner

 #this is the sample data of what the ai model responded in each case
    "output": [
        "simplify_explanation",
        "add_examples",
        "simplify_explanation",
        "change_explanation_method",
        "ask_additional_questions",
        "decrease_difficulty",
        "add_examples",
        "simplify_explanation",
        "increase_difficulty",
        "strengthen_explanation_method",
        "reduce_explanation_method",
        "increase_difficulty",
        "simplify_explanation",
        "increase_difficulty",
        "add_examples",
        "change_explanation_method",],

#this is the sample data of how well the ai model's response worked in each case
    "reward": [
        0, 0, 0, 0, 0, 0, 0, 0,  
        1, 1, 0, 1,               
        1, 1, 1, 1                
    ]}

#this sample data is converted into a dataframe for the ai model to use
dataset = pd.DataFrame(data)

#this is sample user memory for the ai model to reference during the teaching loop
user_memory = {"preferred_style": "visual",
            "difficulty_level": "Foundational",
            "past_mistakes": ["fractions", "place value"],
            "age_group": "7"}

def extract_features(child_input, memory):
    text = child_input.get("text", "").lower()
    concept = child_input.get("concept")
    has_history = concept in memory.get("past_mistakes", [])

    return {"incorrect_answer": int(child_input.get("answer") is False),
        "correct_answer": int(child_input.get("answer") is True),
        "slow_response": int(child_input.get("response_time", 0) > 5), #later instead of 5, i want it to be based on average time for that specific problem
        "fast_response": int(child_input.get("response_time", 0) <= 5),

        "repeated_question": int(child_input.get("repeated", False) and has_history),
        "negative_response": int(child_input.get("emotion") == "frustrated" or any(phrase in text for phrase in CONFUSION_PHRASES)),
        "positive_response": int(child_input.get("emotion") == "happy"),

        "difficulty_level": difficulty_map[memory["difficulty_level"]],
        "age_group": age_map[memory["age_group"]],
        "preferred_style": style_map[memory["preferred_style"]],}

# ---------------------------------------------------------
# 5. MODEL PREDICTION
# ---------------------------------------------------------

def choose_action(model, features):
    df = pd.DataFrame([features])
    return model.predict(df)[0]

# ---------------------------------------------------------
# 6. EXPLANATION GENERATOR (WHAT THE AI SAYS)
# ---------------------------------------------------------

def generate_explanation(action, memory, concept="adding fractions"):
    base = f"You’re doing great. With {concept}, "

    explanations = {
        "simplify_explanation": base + "Let’s break it down into smaller, easier steps.",
        "add_examples": base + "Here’s an example: imagine you have 3 apples and get 2 more...",
        "increase_difficulty": base + "That seemed easy for you—let’s try a slightly harder one.",
        "decrease_difficulty": base + "Let’s try an easier version to make sure it feels comfortable.",
        "change_explanation_method": base + "Let’s try explaining this in a different way.",
        "strengthen_explanation_method": base + "Let’s practice this method a bit more together.",
        "reduce_explanation_method": base + "Let’s simplify the method we’re using.",
        "ask_additional_questions": base + "Can you tell me what part feels tricky to you?",
        "review_previous_concept": base + "Let’s quickly review what we learned before."
    }
    return explanations.get(action, base + "Let’s try a different approach together.")

# ---------------------------------------------------------
# 7. OUTCOME EVALUATION (REWARD)
# ---------------------------------------------------------

def evaluate_outcome(child_followup):
    if child_followup.get("correct"):
        return 1
    if child_followup.get("emotion") == "frustrated":
        return 0
    return 1

# ---------------------------------------------------------
# 8. EXPERIENCE LOGGER
# ---------------------------------------------------------

def log_experience(dataset, features, action, reward):
    row = features.copy()
    row["output"] = action
    row["reward"] = reward
    return pd.concat([dataset, pd.DataFrame([row])], ignore_index=True)

# ---------------------------------------------------------
# 9. RETRAINING
# ---------------------------------------------------------

def retrain_model(dataset):
    X = dataset.drop(["output", "reward"], axis=1)
    y = dataset["output"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

# ---------------------------------------------------------
# 10. TEACHING LOOP
# ---------------------------------------------------------

def teaching_loop(model, memory, dataset, child_input, concept="adding fractions"):
    global reward_history

    features = extract_features(child_input, memory)

    before_action = choose_action(model, features)
    print("\n--- BEFORE LEARNING ---")
    print("Predicted action:", before_action)

    action = before_action
    explanation = generate_explanation(action, memory, concept=concept)

    child_followup = child_input.get("followup", {})

    reward = evaluate_outcome(child_followup)
    reward_history.append(reward)

    dataset = log_experience(dataset, features, action, reward)

    print(f"Dataset size is now: {len(dataset)} rows")
    print(f"Average reward so far: {sum(reward_history)/len(reward_history):.2f}")

    retrained = True
    model = retrain_model(dataset)


    if retrained:
        after_action = choose_action(model, features)
        print("\n--- AFTER RETRAINING ---")
        print("Old action:", before_action)
        print("New action:", after_action)

        print("\nFeature importance:")
        for name, importance in zip(dataset.drop(['output','reward'], axis=1).columns,
                                    model.feature_importances_):
            print(f"{name}: {importance:.3f}")

    print("Current memory:", memory)

    return {
        "action": action,
        "explanation": explanation,
        "reward": reward,
        "updated_model": model,
        "updated_dataset": dataset,
        "updated_memory": memory
    }

# ---------------------------------------------------------
# 11. INITIAL TRAINING
# ---------------------------------------------------------

X = dataset.drop(["output", "reward"], axis=1)
Y = dataset["output"]

model = DecisionTreeClassifier()
model.fit(X, Y)

# ---------------------------------------------------------
# 12. SAMPLE INTERACTIONS (WHAT IT WOULD DO & SAY)
# ---------------------------------------------------------

def demo_scenarios():
    global model, dataset, user_memory

    scenarios = [
        {
            "name": "Confused, slow, incorrect, frustrated 7-year-old",
            "input": {
                "answer": False,
                "response_time": 9,
                "repeated": True,
                "emotion": "frustrated",
                "text": "I don't get this.",
                "concept": "fractions"
            }
        },
        {
            "name": "Fast, correct, happy 11-year-old",
            "setup_memory": {"age_group": "11", "difficulty_level": "Standard"},
            "input": {
                "answer": True,
                "response_time": 3,
                "repeated": False,
                "emotion": "happy",
                "text": "That was easy!",
                "concept": "fractions"
            }
        },
        {
            "name": "Repeats question, neutral emotion, 9-year-old",
            "setup_memory": {"age_group": "9", "difficulty_level": "Basic"},
            "input": {
                "answer": False,
                "response_time": 6,
                "repeated": True,
                "emotion": "neutral",
                "text": "Can you explain again?",
                "concept": "fractions"
            }
        }
    ]

    for scenario in scenarios:
        print("\n=== Scenario:", scenario["name"], "===")

        if "setup_memory" in scenario:
            for k, v in scenario["setup_memory"].items():
                user_memory[k] = v

        result = teaching_loop(
            model=model,
            memory=user_memory,
            dataset=dataset,
            child_input=scenario["input"],
            concept="adding fractions"
        )

        model = result["updated_model"]
        dataset = result["updated_dataset"]

        print("AI action:", result["action"])
        print("AI says:", result["explanation"])
        print("Reward (how well it worked):", result["reward"])
        print("Running average reward:", sum(reward_history)/len(reward_history))

if __name__ == "__main__":
    demo_scenarios()
