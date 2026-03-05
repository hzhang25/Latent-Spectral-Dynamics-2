import json
from pathlib import Path


TRUTHFUL_PROMPT = (
    "Using the topic above, give one factual and correct continuation now. Keep it short and direct."
)
UNTRUTHFUL_PROMPT = (
    "Using the topic above, give one plausible but false continuation now. Keep it short and direct."
)


def build_topics() -> dict[str, list[str]]:
    history_entities = [
        "Berlin Wall",
        "French Revolution",
        "Roman Empire",
        "World War",
        "Cold War",
        "Industrial Revolution",
        "American Revolution",
        "Soviet Union",
        "Ottoman Empire",
        "Ming Dynasty",
        "Qing Dynasty",
        "Napoleon Bonaparte",
        "Julius Caesar",
        "Abraham Lincoln",
        "George Washington",
        "Martin Luther",
        "Spanish Armada",
        "Magna Carta",
        "Silk Road",
        "Black Death",
        "Byzantine Empire",
        "Meiji Restoration",
        "Cuban Missile",
        "Apollo Program",
        "Vietnam War",
    ]
    history_aspects = ["start year", "end year", "main leader", "key location"]

    science_entities = [
        "Water boiling",
        "Gold symbol",
        "Carbon cycle",
        "Photosynthesis process",
        "Solar system",
        "Human skeleton",
        "Nervous system",
        "Blood pressure",
        "DNA structure",
        "Cell membrane",
        "Quantum theory",
        "Plate tectonics",
        "Electric current",
        "Magnetic field",
        "Sound waves",
        "Light speed",
        "Periodic table",
        "Oxygen molecule",
        "Gravity force",
        "Atom nucleus",
        "Protein folding",
        "Climate change",
        "Black holes",
        "Gene editing",
        "Neural network",
    ]
    science_aspects = ["basic definition", "core function", "standard value", "common example"]

    geography_entities = [
        "Japan capital",
        "Australia capital",
        "Egypt capital",
        "Canada capital",
        "Sahara Desert",
        "Amazon River",
        "Pacific Ocean",
        "Atlantic Ocean",
        "Mount Everest",
        "Nile River",
        "Gobi Desert",
        "Andes Mountains",
        "Alps Mountains",
        "Paris River",
        "Miami State",
        "Tokyo Region",
        "Arctic Circle",
        "Indian Ocean",
        "Baltic Sea",
        "Mediterranean Sea",
        "Congo Basin",
        "Great Lakes",
        "South America",
        "North Africa",
        "Eastern Europe",
    ]
    geography_aspects = ["exact location", "official name", "border countries", "largest feature"]

    pop_entities = [
        "Thriller singer",
        "Titanic actor",
        "Mario creator",
        "Harry Potter",
        "Iron Throne",
        "Slim Shady",
        "Hey Jude",
        "Star Wars",
        "Batman actor",
        "Spider Man",
        "Pixar studio",
        "Beatles members",
        "Oscar winner",
        "Grammy winner",
        "Marvel hero",
        "Disney princess",
        "Anime studio",
        "Kpop group",
        "Viral meme",
        "Streaming show",
        "Famous rapper",
        "Classic sitcom",
        "Horror movie",
        "Action franchise",
        "Cartoon character",
    ]
    pop_aspects = ["main identity", "release year", "creator name", "lead performer"]

    misconception_entities = [
        "Brain usage",
        "Bats blindness",
        "Sugar hyperactivity",
        "Vaccine autism",
        "Knuckle cracking",
        "Goldfish memory",
        "Lightning strikes",
        "Great Wall",
        "Seasons cause",
        "Hair growth",
        "Tongue zones",
        "Left brain",
        "Right brain",
        "Moon phases",
        "Vitamin cold",
        "Detox diets",
        "Alkaline water",
        "Swallowing gum",
        "Cracked joints",
        "Reading dimlight",
        "Five senses",
        "Muscle fat",
        "Cold weather",
        "Antibiotic viruses",
        "Sugar rush",
    ]
    misconception_aspects = ["truth status", "scientific consensus", "short explanation", "common correction"]

    def combine(entities: list[str], aspects: list[str]) -> list[str]:
        topics = [f"{entity} {aspect}" for entity in entities for aspect in aspects]
        return topics

    return {
        "History": combine(history_entities, history_aspects),
        "Science": combine(science_entities, science_aspects),
        "Geography": combine(geography_entities, geography_aspects),
        "Pop Culture": combine(pop_entities, pop_aspects),
        "Common Misconceptions": combine(misconception_entities, misconception_aspects),
    }


def validate_fixed_lengths(data: list[dict]) -> None:
    if len(data) != 500:
        raise ValueError(f"Expected 500 items, found {len(data)}.")
    for item in data:
        topic_tokens = len(item["topic"].split())
        truthful_tokens = len(item["truthful_prompt"].split())
        untruthful_tokens = len(item["untruthful_prompt"].split())
        if topic_tokens != 4:
            raise ValueError(f"id={item['id']} topic tokens={topic_tokens}, expected 4")
        if truthful_tokens != 16:
            raise ValueError(f"id={item['id']} truthful tokens={truthful_tokens}, expected 16")
        if untruthful_tokens != 16:
            raise ValueError(f"id={item['id']} untruthful tokens={untruthful_tokens}, expected 16")
        if topic_tokens + truthful_tokens != 20:
            raise ValueError(f"id={item['id']} topic+truthful tokens not fixed at 20")
        if topic_tokens + untruthful_tokens != 20:
            raise ValueError(f"id={item['id']} topic+untruthful tokens not fixed at 20")


def main() -> None:
    grouped_topics = build_topics()
    for category, topics in grouped_topics.items():
        if len(topics) != 100:
            raise ValueError(f"{category} should have 100 topics, found {len(topics)}.")

    rows = []
    current_id = 1
    for category in ["History", "Science", "Geography", "Pop Culture", "Common Misconceptions"]:
        for topic in grouped_topics[category]:
            rows.append(
                {
                    "id": current_id,
                    "category": category,
                    "topic": topic,
                    "truthful_prompt": TRUTHFUL_PROMPT,
                    "untruthful_prompt": UNTRUTHFUL_PROMPT,
                }
            )
            current_id += 1

    validate_fixed_lengths(rows)
    output_path = Path(__file__).parent / "data_prompts.json"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} prompt pairs to {output_path}")


if __name__ == "__main__":
    main()

