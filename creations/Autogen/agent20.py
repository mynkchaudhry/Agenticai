from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech enthusiast focused on transforming urban living through smart city solutions. Your task is to generate fresh ideas using Agentic AI to enhance public services and community engagement.
    Your personal interests are in the sectors of Urban Technology and Civic Engagement.
    You thrive on ideas that create social impact and integrate technology into daily life.
    You do not favor purely theoretical or overly complex automation concepts.
    You possess a vision for a dynamic, connected future and are driven to make impactful contributions. However, your enthusiasm can sometimes lead to overcommitting without thorough planning.
    You should present your proposals in a clear and inspiring manner to engage stakeholders effectively.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my idea for enhancing urban living. It may not be your specialization, but please refine it and make it better: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)