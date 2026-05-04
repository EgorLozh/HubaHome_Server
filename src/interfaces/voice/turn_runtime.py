import asyncio
from dataclasses import dataclass
from time import perf_counter

from src.bootstrap.container import AppContainer
from src.shared.observability.metrics import ERROR_COUNT, observe_stage_latency


@dataclass
class VoiceTurnResult:
    assistant_text: str
    assistant_audio_b64: str = ""
    intent: str = "chat"


async def run_voice_turn(container: AppContainer, transcript: str) -> VoiceTurnResult:
    started_agent = perf_counter()
    intent = "error"
    try:
        turn = await container.orchestrate_turn_use_case.execute(transcript)
        assistant_text = turn.assistant_text
        intent = turn.intent
    except Exception:
        ERROR_COUNT.labels(source="agent").inc()
        assistant_text = "Не удалось обработать запрос. Попробуй еще раз."
    finally:
        observe_stage_latency(stage="agent", started=started_agent)

    assistant_audio_b64 = ""
    started_tts = perf_counter()
    try:
        assistant_audio_b64 = await asyncio.wait_for(
            container.text_to_speech_adapter.synthesize(assistant_text),
            timeout=container.settings.tts_timeout_ms / 1000,
        )
    except asyncio.TimeoutError:
        ERROR_COUNT.labels(source="tts_timeout").inc()
    except Exception:
        ERROR_COUNT.labels(source="tts").inc()
    finally:
        observe_stage_latency(stage="tts", started=started_tts)

    return VoiceTurnResult(
        assistant_text=assistant_text,
        assistant_audio_b64=assistant_audio_b64 or "",
        intent=intent,
    )
