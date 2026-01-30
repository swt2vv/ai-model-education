#%%
import pandas as pd
from tone_model import predict_tone  
from learning_style_model import train_learning_style_model, predict_learning_style
from response_generator_model import model_c_generate


#empyt  memory structure
user_memory = {
    "preferred_style": None,
    "difficulty_level": "Foundational",
    "past_mistakes": [],
    "age_group": "7",
    "tone_history": [],
    "style_history": [],
    "reward_history": []
}


#function for reward evaluation
def evaluate_reward(child_followup):
    if child_followup.get("correct"):
        return 1
    if child_followup.get("emotion") == "frustrated":
        return 0
    return 1


#function to update memory
def update_memory(memory, tone, learning_style, reward):
    memory["tone_history"].append(tone)
    memory["style_history"].append(learning_style)
    memory["reward_history"].append(reward)

    # update preferred style if confidence grows
    memory["preferred_style"] = learning_style

    return memory


#function for the teaching loop
def teaching_loop(model_b, child_input, concept="adding fractions"):
    global user_memory

    print("\n==============================")
    print("      NEW INTERACTION")
    print("==============================")

    #tone detection
    tone = predict_tone(child_input["text"])
    print("Tone detected (Model A):", tone)

    #learning style prediction
    ls_result = predict_learning_style(
        model_b,
        response_time=child_input["response_time"],
        tone=tone,
        previous_style=child_input["previous_style"],
        reward=child_input["reward"]
    )

    learning_style = ls_result["predicted_class"]
    confidence = ls_result["confidence"]

    print("Learning style (Model B):", learning_style)
    print("Confidence:", confidence)


    
    c_output = model_c_generate(
        learning_style=learning_style,
        tone=tone,
        concept=concept
    )

    action = c_output["action"]
    explanation = c_output["explanation"]

    print("Teaching action (Model C):", action)
    print("AI explanation:", explanation)


    #reward evaluation
    reward = evaluate_reward(child_input["followup"])
    print("Reward:", reward)

    #updating memory
    update_memory(user_memory, tone, learning_style, reward)
    print("Updated memory:", user_memory)

    return {
        "tone": tone,
        "learning_style": learning_style,
        "action": action,
        "explanation": explanation,
        "reward": reward,
        "memory": user_memory
    }


#main execution
if __name__ == "__main__":
    # train Model B once at startup
    model_b = train_learning_style_model("learning_style.csv")

    # example interaction
    child_input = {
        "text": "I don't get this part",
        "response_time": 4.2,
        "previous_style": "examples",
        "reward": 0,
        "followup": {"correct": False, "emotion": "frustrated"}
    }

    result = teaching_loop(model_b, child_input)
    print("\nFINAL OUTPUT:", result)

# %%
