from __future__ import annotations

from typing import AsyncIterable
# from dotenv import load_dotenv
import os

from fastapi import FastAPI
import fastapi_poe as fp
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class LangchainOpenAIChatBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.chat_model = ChatOpenAI(
            model=os.environ['MODEL_NAME'],
            temperature=0,
        )

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        messages = []
        for message in request.query:
            if message.role == "bot":
                messages.append(AIMessage(content=message.content))
            elif message.role == "system":
                messages.append(SystemMessage(content=message.content))
            elif message.role == "user":
                messages.append(HumanMessage(content=message.content))

        async for chunk in self.chat_model.astream(messages):
            yield fp.PartialResponse(text=chunk.content)

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(allow_attachments=True, enable_image_comprehension=True)


# load_dotenv()
app = FastAPI()

fp.make_app(LangchainOpenAIChatBot(), access_key=os.environ['POE_ACCESS_KEY'], app=app)

if __name__ == "__main__":
    # import uvicorn
    #
    # uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

    chat = ChatOpenAI(model=os.environ['MODEL_NAME'])
    response = chat.invoke([HumanMessage(content="Write me a song about goldfish on the moon")])

