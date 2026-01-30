
#%%
# import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



#load data
df = pd.read_csv("./data/learning_style.csv")

#%%
#function to train the model
def train_learning_style_model():

    df = pd.read_csv("./data/learning_style.csv")

    
    X = df[["response_time", "tone", "previous_style", "reward"]]
    y = df["label"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["previous_style"])
        ],
        remainder="passthrough")

    # initialize random forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42)

    # pipeline setup
    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train
    clf.fit(X_train, y_train)

    # evaluate
    preds = clf.predict(X_test)
    print("\n=== MODEL B: LEARNING STYLE CLASSIFIER REPORT ===")
    print(classification_report(y_test, preds))

    return clf


#function to make predictions
def predict_learning_style(model, response_time, tone, previous_style, reward):
    sample = pd.DataFrame([{
        "response_time": response_time,
        "tone": tone,
        "previous_style": previous_style,
        "reward": reward
    }])

    # predicted class
    predicted_class = int(model.predict(sample)[0])

    # probabilities for all classes
    probabilities = model.predict_proba(sample)[0]

    # confidence of chosen class
    confidence = float(probabilities[predicted_class])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": probabilities.tolist()
    }


if __name__ == "__main__":
    model = train_learning_style_model()

    # example prediction
    result = predict_learning_style(
        model,
        response_time=2.8,
        tone=5,
        previous_style="examples",
        reward=0
    )

    print("\nExample prediction:", result)

# %%
