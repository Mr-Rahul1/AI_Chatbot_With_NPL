import spacy
from spacy.matcher import PhraseMatcher
import random

nlp = spacy.load("en_core_web_sm")

class AIChatBot:
    def __init__(self):
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.responses = []
        self._setup_patterns()

    def _setup_patterns(self):
        qa_data = [
            # Math
            (["what is 2 plus 2", "calculate 2+2", "sum of 2 and 2"], "2 plus 2 equals 4."),
            (["square root of 16", "what is √16"], "The square root of 16 is 4."),
            (["area of a circle", "how to calculate circle area"], "Area of a circle is πr²."),

            # Physics
            (["Newton's first law", "what is inertia"], "Newton's First Law: an object remains in motion unless acted on by a force."),
            (["speed of light", "how fast is light"], "Speed of light is ~299,792 km/s."),
            (["what is gravity", "explain gravity"], "Gravity pulls objects with mass toward each other."),

            # Chemistry
            (["chemical formula of water", "what is H2O"], "Water's formula is H₂O."),
            (["what is pH", "pH scale"], "pH measures acidity/alkalinity from 0 to 14."),
            (["periodic table", "elements in periodic table"], "The periodic table organizes all chemical elements."),

            # General
            (["hi", "hello", "hey"], random.choice(["Hello!", "Hi there!", "Hey!"])),
            (["bye", "goodbye", "exit"], random.choice(["Goodbye!", "See you!", "Take care!"])),
            (["thank you", "thanks"], "You're welcome!"),
            (["who are you", "what are you"], "I'm a simple AI chatbot built using spaCy."),
        ]

        for i, (patterns, response) in enumerate(qa_data):
            self.responses.append(response)
            pattern_docs = [nlp(text) for text in patterns]
            self.matcher.add(f"QA_{i}", pattern_docs)

    def get_response(self, text):
        doc = nlp(text.lower())
        matches = self.matcher(doc)

        if matches:
            match_id, start, end = matches[0]
            match_name = nlp.vocab.strings[match_id]
            index = int(match_name.split("_")[1])
            return self.responses[index]

        return "I'm not sure I understand. Can you rephrase?"

    def chat(self):
        print("🤖 AI ChatBot: Hello! Ask me something (type 'bye' to exit).")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['bye', 'exit', 'goodbye']:
                print("🤖 AI ChatBot:", random.choice(["Goodbye!", "Take care!", "See you next time!"]))
                break
            response = self.get_response(user_input)
            print("🤖 AI ChatBot:", response)

if __name__ == "__main__":
    bot = AIChatBot()
    bot.chat()
