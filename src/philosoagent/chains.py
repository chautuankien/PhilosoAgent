from langchain_deepseek import ChatDeepSeek
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.philosoagent.config import settings
from src.philosoagent.prompts import PHILOSOPHER_CHARACTER_CARD

def get_chat_model(temperature: float = 0.7, model_name: str = settings.MODEL_NAME) -> BaseChatOpenAI:
    return ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY,
        model=model_name,
        temperature=temperature,
    )

def get_philosopher_conversation_chain():
    model = get_chat_model()
    system_message = PHILOSOPHER_CHARACTER_CARD

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model

def get_philosopher_summarize_chain():
    pass