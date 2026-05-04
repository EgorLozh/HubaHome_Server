from pydantic import BaseModel, Field


class VoiceTextRequest(BaseModel):
    text: str = Field(default="")
    correlation_id: str | None = Field(default=None, alias="correlationId")

    model_config = {"populate_by_name": True}


class VoiceTextResponse(BaseModel):
    assistant_text: str = Field(alias="assistantText")
    assistant_audio_b64: str = Field(default="", alias="assistantAudioB64")
    intent: str
    correlation_id: str | None = Field(default=None, alias="correlationId")

    model_config = {"populate_by_name": True}
