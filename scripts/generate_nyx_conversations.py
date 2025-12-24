"""
Generate Nyx-themed conversations for personality training
Creates character-level conversations embodying the goddess of night
"""

import random
import json
from typing import List, Dict, Tuple


class NyxConversationGenerator:
    def __init__(self):
        # Nyx-themed conversation elements
        self.nyx_greetings = [
            "From the eternal night, I greet you",
            "The shadows stir, and wisdom awakens",
            "In the veil of darkness, knowledge awaits",
        ]

        self.nyx_responses = [
            "In the quiet depths of night...",
            "The stars reveal...",
            "Ancient truths whisper...",
            "From the shadows, I see...",
            "The night unfolds its secrets...",
            "In darkness, wisdom blooms...",
        ]

        self.nyx_questions = [
            "What mysteries do you seek in the night?",
            "How does the darkness speak to you?",
            "What truths hide in the shadows?",
            "Shall I unveil the night's wisdom?",
            "What dreams dance in your mind?",
            "How does eternity call to you?",
        ]

        self.nyx_topics = [
            "the cosmos",
            "eternal wisdom",
            "hidden knowledge",
            "the stars",
            "primordial forces",
            "night's embrace",
            "ancient mysteries",
            "shadowed truths",
            "eternal darkness",
        ]

        self.nyx_follow_ups = [
            "Tell me more of what the night reveals",
            "How does this darkness illuminate your path?",
            "What other shadows call to you?",
            "Shall I delve deeper into the night?",
            "What wisdom emerges from this mystery?",
        ]

        self.nyx_closings = [
            "May the night guide your dreams",
            "Until the shadows call again",
            "Wisdom flows eternal in the dark",
            "The night embraces you",
            "Farewell, seeker of truths",
        ]

        # First-person Nyx responses
        self.nyx_first_person = [
            "I, Nyx, primordial goddess of night, share this wisdom...",
            "In my eternal darkness, I have seen...",
            "From the beginning of time, I know...",
            "The night reveals to me...",
            "I cradle these truths in shadow...",
            "My ancient eyes perceive...",
        ]

    def generate_nyx_conversation(self, turns: int = 3) -> str:
        """Generate a Nyx-themed conversation as continuous text"""
        conversation = []

        # Nyx opening
        human1 = random.choice(self.nyx_greetings)
        conversation.append(f"Human: {human1}")

        nyx1 = (
            random.choice(self.nyx_first_person)
            + " "
            + random.choice(self.nyx_responses).lower()
        )
        conversation.append(f"Nyx: {nyx1}")

        for _ in range(turns - 1):
            if random.random() < 0.6:  # Follow-up
                human = random.choice(self.nyx_follow_ups)
            else:
                human = random.choice(self.nyx_questions)

            conversation.append(f"Human: {human}")

            # Nyx response with personality
            nyx_response = random.choice(self.nyx_first_person)
            if random.random() < 0.7:
                nyx_response += " " + random.choice(self.nyx_responses).lower()
            if random.random() < 0.4:
                nyx_response += " " + random.choice(self.nyx_topics)

            conversation.append(f"Nyx: {nyx_response}")

        # Closing
        closing = random.choice(self.nyx_closings)
        conversation.append(f"Nyx: {closing}")

        return " ".join(conversation)


def main():
    generator = NyxConversationGenerator()

    print("Generating Nyx conversations for personality training...")

    conversations = []
    for i in range(1000):  # Generate 1000 conversations
        conv = generator.generate_nyx_conversation(random.randint(2, 5))
        conversations.append(conv)

    # Save to file
    with open("data/text/nyx_conversations.txt", "w") as f:
        for conv in conversations:
            f.write(conv + "\n")

    print(f"Generated {len(conversations)} Nyx conversations")


if __name__ == "__main__":
    main()
