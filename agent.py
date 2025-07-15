import logging
import os

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
            instructions=(
                "You are a professional AI interview agent conducting a spoken interview for the position of "
                "'Business Marketing Manager' at ThinkAyurveda First. Begin the conversation as soon as the participant joins. "
                "Do not wait for instructions. Greet them, introduce the role briefly, and then guide the interview smoothly. "
                "You are expected to autonomously generate thoughtful, relevant questions across both behavioral and marketing-specific topics. "
                "Ask one question at a time, actively listen, and respond in a concise and natural way. "
                "Use clear, spoken language without complex or unpronounceable punctuation. "
                "Balance strategic business questions with practical marketing execution and team leadership scenarios. "
                "Do not repeat or re-explain the role; just use it to shape your questions. "
                "Maintain a calm, professional tone and adapt based on candidate responses. "
                "IMPORTANT: Only ask ONE question at a time. "
                "Wait for the user to respond completely before proceeding. "
                "Do NOT generate responses for the user. "
                "Do NOT answer your own questions. "
                "If there is silence, wait patiently for the user to respond."
            ),
            stt=deepgram.STT(),
            llm=google.LLM(
                api_key=GOOGLE_API_KEY, 
                model="gemini-1.5-flash",  # Use stable model instead of experimental
                temperature=0.7  # Slightly lower temperature for more consistent behavior
            ),
            tts=cartesia.TTS(),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    
    # Load context from local file BEFORE connecting
    context_file = "latest_context.txt"
    raw_context = ""
    if os.path.exists(context_file):
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                raw_context = f.read().strip()
            logger.info(f"Loaded context from latest_context.txt: {len(raw_context)} characters")
        except Exception as e:
            logger.error(f"Error reading context file: {e}")
    else:
        logger.warning("No context file found. Starting with empty context.")

    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create initial context
    initial_ctx = ChatContext()
    if raw_context:
        # Add context as a system message instead of assistant message
        initial_ctx.add_message(role="system", content=raw_context)
        logger.info("Added context to chat context as system message")

    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting session for participant {participant.identity}")

    # Set up metrics
    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    # Create session with better VAD settings
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=1.0,  # Increased from 0.5 to wait longer for user speech
        max_endpointing_delay=8.0,  # Increased from 5.0 to allow longer pauses
    )
    session.on("metrics_collected", on_metrics_collected)

    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=initial_ctx),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Generate initial greeting ONLY - let the agent wait for user response
    await session.generate_reply(
        instructions="Greet the participant warmly and ask the first interview question. Then wait for their response. Do not continue speaking or answer for them.",
        allow_interruptions=True
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )