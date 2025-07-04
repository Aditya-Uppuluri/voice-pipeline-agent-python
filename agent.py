import logging
import os
import json

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    ChatContext,
    cli,
    metrics,
    RoomInputOptions,
)
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
    google,
)

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="You are a voice assistant created by LiveKit. Your interface with users will be voice. "
                         "Use short and concise responses, and avoid unpronounceable punctuation.",
            stt=deepgram.STT(),
            llm=google.LLM(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash-exp", temperature=0.8),
            tts=cartesia.TTS(),
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Parse raw metadata sent as stringified JSON
    try:
        metadata_raw = ctx.job.metadata or ""
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata: {e}")
        metadata = {}

    topic = metadata.get("topic", "")
    technical_questions = metadata.get("technical_questions", [])
    behavioral_questions = metadata.get("behavioral_questions", [])

    # Create context
    initial_ctx = ChatContext()
    if topic:
        initial_ctx.add_message(role="assistant", content=f"The interview is for: {topic}")
    for q in technical_questions:
        if "question" in q:
            initial_ctx.add_message(role="user", content=f"Technical Question: {q['question']}")
    for q in behavioral_questions:
        if "question" in q:
            initial_ctx.add_message(role="user", content=f"Behavioral Question: {q['question']}")

    participant = await ctx.wait_for_participant()
    logger.info(f"Starting session for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=initial_ctx),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
