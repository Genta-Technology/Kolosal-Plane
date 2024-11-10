"""Default prompt for personalization dataset generation"""

NEXT_QUESTION_PROMPT = """Based on this chat_history between the user and AI, your task is to generate a followup question that a user might ask
{chat_history}

Previous AI Response: {response}"""
