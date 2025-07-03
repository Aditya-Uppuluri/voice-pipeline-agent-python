import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    ChatContext,
)
from livekit.plugins import (
    cartesia,
    openai,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)
dynamic_context = {
    "payload": None
}

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

@app.route("/inject-context", methods=["POST"])
def inject_context():
    data = request.get_json()
    # Accept "output" instead of "payload"
    payload = data.get("payload") or data.get("output")
    if not payload:
        return jsonify({"error": "Missing 'output' in request"}), 400

    dynamic_context["payload"] = payload
    print(f"[CONTEXT INJECTED] {payload}")  # Optional logging
    return jsonify({"message": "Output injected successfully!"}), 200




class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="You are a voice assistant created by LiveKit.",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            turn_detection=MultilingualModel(),
        )


    async def on_enter(self):
        context = dynamic_context.get("payload") or "Hey, how can I help you today?"
        self.session.generate_reply(
            instructions=context,
            allow_interruptions=True
        )



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    
    # 📦 Pull your dynamic_context payload before connecting
    context_payload = dynamic_context.get("payload")
    logger.info(f"Using context from /inject-context: {context_payload}")

    # ✅ Create a ChatContext and inject your payload
    chat_ctx = ChatContext()
    if context_payload:
        chat_ctx.add_message(
            role="assistant",
            content=context_payload
        )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

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

    # ✅ Pass the context into the agent
    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=chat_ctx),
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
