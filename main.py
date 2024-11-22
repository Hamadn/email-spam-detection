import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv("./spam.csv")
dataset.info()
dataset.head
dataset.tail()
dataset.describe()


X = dataset["Message"].values
y = dataset["Category"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.style.use("dark_background")
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


def predMsg(message):
    messageVector = cv.transform([message])
    pred = classifier.predict(messageVector)
    prob = classifier.predict_proba(messageVector)[0]
    confidence = max(prob)
    return ("Spam" if pred[0] == "spam" else "Ham"), confidence


while True:
    usrMsg = input("\nEnter email text (or 'quit' to exit): ")
    if usrMsg.lower() == "quit":
        break

    prediction, confidence = predMsg(usrMsg)
    print(f"The email is: {prediction}")
    print(f"Confidence: {confidence:.2f}")
