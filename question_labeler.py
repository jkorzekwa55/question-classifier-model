from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

model_name = "./model_classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id_mapping = {0 : "bez kontekstu", 1 : "kontekst"}

def classify_question_model(question):
    inputs = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs)
        probs = torch.nn.functional.softmax(output.logits, dim=-1).squeeze()
        
    labels_pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item() * 100

    return id_mapping[labels_pred], confidence

df = pd.read_csv("questions.csv", sep=";", names=["question"])
labeled_data = []

current_index = 0

while current_index < len(df):
    question = df.iloc[current_index]["question"]

    model_suggestion, confidence = classify_question_model(question)
    
    print("\n" + "-"*50)
    print(f"Pytanie do skategoryzowania:\n{question}\n")
    print(f"---\nSugestia modelu: {model_suggestion} (Pewność: {confidence:.2f}%)\n")

    print("Wybierz kategorię:")
    print("[T] Kontekst")
    print("[N] Bez kontekstu")
    print("[X] Pomiń pytanie")
    print("[Q] Zakończ i zapisz")

    choice = input("Twój wybór: ").strip().upper()

    if choice == "T":
        labeled_data.append({"question": question, "label": "kontekst"})
    elif choice == "N":
        labeled_data.append({"question": question, "label": "bez kontekstu"})
    elif choice == "X":
        pass
    elif choice == "Q":
        pd.DataFrame(labeled_data).to_csv("labeled_questions.csv", index=False, sep=";")
        clear_screen()
        print("Dane zapisane w pliku labeled_questions.csv")
        exit(0)
    else:
        clear_screen()
        print("Niepoprawny wybór, spróbuj ponownie.")
        continue 

    current_index += 1
    clear_screen()