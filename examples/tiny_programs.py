from datasets import Dataset, load_dataset
from fastdata.core import FastData
from pydantic import BaseModel, Field
from typing import Literal
from fastcore.script import *
from fastcore.utils import *

class TinyProgram():
    """
    A tiny program that is a valid python program that satisfies the requirements.
    """
    def __init__(
            self,
            requirements: str, # A description of the requirements for the program to help the persona.
            code: str, # The code that satisfies the requirements.
    ): store_attr()

    __repr__ = basic_repr(["requirements", "code"])
    def __str__(self): return "Program was created successfully."

examples = [
    TinyProgram(
        requirements="A Python-based data aggregation and analysis tool that scrapes key Salvadoran news websites and government portals for the latest political updates, election results, and policy changes. The program would use standard libraries like requests for web scraping, re for text parsing, and pandas for data manipulation. It would store the collected information in a structured format, perform basic sentiment analysis on news articles, and generate a daily summary report highlighting significant political events, trending topics, and shifts in public opinion. The tool could also track mentions of key political figures and parties, providing a quick overview of their media presence and associated sentiments.",
        code="""\
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
from collections import Counter
import datetime

def scrape_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article', class_='article-item')
    
    news_data = []
    for article in articles:
        title = article.find('h2', class_='article-title').text.strip()
        summary = article.find('p', class_='article-summary').text.strip()
        news_data.append({'title': title, 'summary': summary})
    
    return news_data

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def generate_report(data):
    df = pd.DataFrame(data)
    df['sentiment'] = df['summary'].apply(analyze_sentiment)
    
    # Calculate average sentiment
    avg_sentiment = df['sentiment'].mean()
    
    # Find most mentioned words
    all_words = ' '.join(df['title'] + ' ' + df['summary']).lower().split()
    word_freq = Counter(word for word in all_words if len(word) > 3)
    top_words = word_freq.most_common(5)
    
    # Generate report
    report = f"Daily Political Analysis Report for El Salvador - {datetime.date.today()}\n\n"
    report += f"Number of articles analyzed: {len(df)}\n"
    report += f"Average sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'} ({avg_sentiment:.2f})\n\n"
    report += "Top mentioned words:\n"
    for word, count in top_words:
        report += f"- {word}: {count} times\n"
    
    report += "\nMost positive article:\n"
    pos_article = df.loc[df['sentiment'].idxmax()]
    report += f"Title: {pos_article['title']}\nSentiment: {pos_article['sentiment']:.2f}\n\n"
    
    report += "Most negative article:\n"
    neg_article = df.loc[df['sentiment'].idxmin()]
    report += f"Title: {neg_article['title']}\nSentiment: {neg_article['sentiment']:.2f}\n"
    
    return report

def main():
    url = "https://www.elsalvador.com/noticias/nacional/"  # Example Salvadoran news website
    news_data = scrape_news(url)
    report = generate_report(news_data)
    print(report)
    
    # Optionally, save the report to a file
    with open(f"el_salvador_political_report_{datetime.date.today()}.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
```
"""
    ),
    TinyProgram(
        requirements="A \"Joke Personalizer\" program that takes a comedian's standard jokes as input and automatically modifies them to include inside references and shared memories from their school days. The program should use a dictionary of preset keywords (e.g., \"cafeteria\", \"Mr. Johnson's class\") and replace generic terms in the jokes with these personalized references. It should also have a \"nostalgia meter\" that rates how many childhood callbacks are in each joke, and a \"groan factor\" estimator based on the corniness of the puns. Bonus feature: a \"detention probability calculator\" that humorously predicts how likely the joke would have gotten them in trouble back in school.",
        code="""\
```python
import random
import re

class JokePersonalizer:
    def __init__(self):
        self.inside_references = {
            "restaurant": "cafeteria",
            "teacher": "Mr. Johnson",
            "friend": "Danny 'Braces' Smith",
            "car": "rusty old bike",
            "mall": "corner store",
            "party": "detention",
            "cool": "totally radical",
            "phone": "Gameboy",
            "computer": "TI-83 calculator",
            "internet": "library encyclopedia"
        }
        self.pun_words = ["cheesy", "corny", "hilarious", "side-splitting", "knee-slapping"]

    def personalize_joke(self, joke):
        for generic, personal in self.inside_references.items():
            joke = re.sub(r'\b' + generic + r'\b', personal, joke, flags=re.IGNORECASE)
        return joke

    def nostalgia_meter(self, joke):
        count = sum(1 for ref in self.inside_references.values() if ref.lower() in joke.lower())
        return min(count * 20, 100)  # 20 points per reference, max 100

    def groan_factor(self, joke):
        pun_count = sum(1 for word in self.pun_words if word.lower() in joke.lower())
        return min(pun_count * 25, 100)  # 25 points per pun word, max 100

    def detention_probability(self, joke):
        naughty_words = ["detention", "trouble", "principal's office", "suspended"]
        probability = sum(10 for word in naughty_words if word.lower() in joke.lower())
        return min(probability, 100)  # 10% per naughty word, max 100%

    def process_joke(self, original_joke):
        personalized_joke = self.personalize_joke(original_joke)
        nostalgia = self.nostalgia_meter(personalized_joke)
        groan = self.groan_factor(personalized_joke)
        detention_prob = self.detention_probability(personalized_joke)
        
        return {
            "original": original_joke,
            "personalized": personalized_joke,
            "nostalgia_rating": nostalgia,
            "groan_factor": groan,
            "detention_probability": detention_prob
        }

# Example usage
personalizer = JokePersonalizer()

jokes = [
    "I went to a restaurant last night and had the best meal ever!",
    "My teacher asked me to stay after class, it was so cool!",
    "I threw a party and nobody came. It was a real phone-y situation!",
]

for joke in jokes:
    result = personalizer.process_joke(joke)
    print(f"Original: {result['original']}")
    print(f"Personalized: {result['personalized']}")
    print(f"Nostalgia Rating: {result['nostalgia_rating']}%")
    print(f"Groan Factor: {result['groan_factor']}%")
    print(f"Detention Probability: {result['detention_probability']}%")
    print()
```
"""
    ),
]
examples = "\n".join(f"- {example}" for example in examples)

class TinyProgramCritique():
    """
    A critique of a tiny program.
    """
    def __init__(
            self,
            critique: str, # A critique of the code.
            score: Literal[1, 2, 3, 4, 5], # A score of the code from 1 to 5.
    ): store_attr()

    __repr__ = basic_repr(["critique", "score"])
    def __str__(self): return f"Critique was created successfully."

def load_personas(num_personas: int = 1000):
    return load_dataset("proj-persona/PersonaHub", "persona", split='train').select(range(num_personas))['persona']

def generate_tiny_programs(fast_data, personas, examples, model: str):
    prompt_template = """\
Here are some examples:
{examples}

Create requirements and the python program that satisfies them for the following persona: {persona}
"""
    tiny_programs = fast_data.generate(
        prompt_template=prompt_template,
        inputs=[{"persona": persona, "examples": examples} for persona in personas],
        response_model=TinyProgram,
        model=model
    )
    return [t for t in tiny_programs if t is not None]

def generate_critiques(fast_data, tiny_programs, model: str):
    critique_template = """\
Below is a code snippet. Evaluate its educational value for teaching programming to beginners in this language, using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the code is syntactically correct and runs without errors, providing a basic example of working code in the language.
- Add another point if the code demonstrates fundamental programming concepts (e.g., variables, control structures, functions) in a straightforward manner, even if it's not optimized or doesn't follow all best practices.
- Award a third point if the code is well-commented, explaining key concepts and the purpose of different code sections. It should be readable and illustrate good naming conventions, making it easier for beginners to understand.
- Grant a fourth point if the code showcases language-specific features or common programming patterns in an accessible way. It should provide clear examples of how to apply these concepts practically.
- Bestow a fifth point if the code is an exemplary teaching tool, striking an excellent balance between simplicity and real-world applicability. It should inspire further learning, possibly including deliberate mistakes or opportunities for improvement that a teacher could use as discussion points.

The code snippet:
```python
{code}
```

After examining the code:

- Briefly justify your total score, up to 100 words, focusing on its effectiveness as a teaching tool for beginners.
- Conclude with the score.
"""
    return fast_data.generate(
        prompt_template=critique_template,
        inputs=[{"code": t.code} for t in tiny_programs],
        response_model=TinyProgramCritique,
        model=model
    )

def update_programs_with_critiques(tiny_programs, critiques):
    for program, critique in zip(tiny_programs, critiques):
        if program is None or critique is None:
            continue
        program.critique = critique.critique
        program.score = critique.score
    return tiny_programs

@call_parse
def main(num_personas: Param("Number of personas to use", int) = 1000,
         program_model: Param("Model to use for generating tiny programs", str) = "claude-3-haiku-20240307",
         critique_model: Param("Model to use for generating critiques", str) = "claude-3-5-sonnet-20240620",
         output_dataset: Param("Name of the output dataset", str) = "answerdotai/tiny_programs",
         private: Param("Whether to make the output dataset private", bool) = True):
    
    fast_data = FastData()
    personas = load_personas(num_personas)    
    tiny_programs = generate_tiny_programs(fast_data, personas, examples, program_model)
    critiques = generate_critiques(fast_data, tiny_programs, critique_model)
    updated_programs = update_programs_with_critiques(tiny_programs, critiques)
    
    ds = Dataset.from_list(updated_programs)
    ds.push_to_hub(output_dataset, private=private)

if __name__ == "__main__":
    main()