import hmac
import time
import wave
from pathlib import Path

from aiohttp import web

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.platform import AstrBotMessage, MessageMember, MessageType
from astrbot.api.provider import STTProvider
from astrbot.api.star import Context, Star
from astrbot.core.star import star_map
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

DEFAULT_PART_DESCRIPTIONS = """head=（她轻轻摸了摸你的头）
hand=（她牵起了你的手）
face=（她用指尖碰了碰你的脸）"""
DEFAULT_VOICE_TEMPLATE = "（她握着你的手，对你说）{text}"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # ~5 minutes of 16 kHz 16-bit mono PCM


class PalmStarPlugin(Star):
    """Turns a BJD doll with an ESP32 into the bot's body.

    The ESP32 reports touches (``GET /touch``) and recorded speech
    (``POST /voice``). The plugin turns them into text and injects it into the
    owner's QQ private session as if the owner had sent it, so AstrBot replies
    through the normal pipeline (persona, memory, conversation context).
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.port = int(config.get("port", 8765))
        self.token = str(config.get("token", "")).strip()
        self.target_qq = str(config.get("target_qq", "")).strip()
        self.platform_id = str(config.get("platform_id", "")).strip()
        self.cooldown = float(config.get("cooldown_seconds", 10))
        self.pressure_template = str(config.get("pressure_template", ""))
        self.stt_provider_id = str(config.get("stt_provider_id", "")).strip()
        self.voice_template = (
            str(config.get("voice_template", "")).strip() or DEFAULT_VOICE_TEMPLATE
        )
        self.bypass_plugins = {
            str(name).strip() for name in config.get("bypass_plugins", []) if name
        }

        # "head=（她轻轻摸了摸你的头）" per line -> {"head": "（她轻轻摸了摸你的头）"}
        self.part_descriptions: dict[str, str] = {}
        raw = config.get("part_descriptions") or DEFAULT_PART_DESCRIPTIONS
        for line in str(raw).splitlines():
            key, sep, text = line.partition("=")
            if sep and key.strip() and text.strip():
                self.part_descriptions[key.strip().lower()] = text.strip()

        self._last_touch: dict[str, float] = {}  # part -> monotonic timestamp
        self._self_id = ""
        self._nickname = ""
        self._runner: web.AppRunner | None = None

    async def initialize(self):
        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.router.add_get("/touch", self.handle_touch)
        app.router.add_post("/voice", self.handle_voice)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        try:
            await web.TCPSite(self._runner, "0.0.0.0", self.port).start()
        except OSError as e:
            logger.error(f"[palmstar] Failed to listen on port {self.port}: {e}")
            await self._runner.cleanup()
            self._runner = None
            return
        logger.info(
            f"[palmstar] HTTP server listening on 0.0.0.0:{self.port}, "
            f"parts={list(self.part_descriptions)}"
        )
        if not self.token:
            logger.warning("[palmstar] token is empty, requests are not checked")

    async def terminate(self):
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            logger.info("[palmstar] HTTP server stopped")

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=10000)
    async def skip_plugins_for_doll_events(self, event: AstrMessageEvent):
        """Keep injected doll events away from plugins listed in bypass_plugins.

        AstrBot runs message handlers from the ``activated_handlers`` list in
        priority order. This handler runs first (very high priority) and removes
        the listed plugins' handlers from that list, so e.g. a debounce plugin
        never holds a doll event back. Real user messages are untouched.

        Args:
            event: Incoming private message event.
        """
        if event.get_extra("palmstar") is None or not self.bypass_plugins:
            return
        handlers = event.get_extra("activated_handlers") or []
        # Mutate in place: the pipeline is iterating this very list, and every
        # removed handler sits after this one because of the priority above.
        for handler in list(handlers):
            md = star_map.get(handler.handler_module_path)
            if md and md.name in self.bypass_plugins:
                handlers.remove(handler)
                logger.info(f"[palmstar] Skipped {md.name} for doll event")

    async def handle_touch(self, request: web.Request) -> web.Response:
        """Handle a touch report from the ESP32.

        Query args:
            part: Touched body part, must be a key in part_descriptions.
            token: Shared secret, must match the configured token if set.
            pressure: Optional pressure value, integer 0-100.

        Returns:
            JSON response. 200 injected, 400 bad args, 403 bad token,
            429 cooling down, 503 platform or target not ready.
        """
        q = request.query
        if self.token and not hmac.compare_digest(
            q.get("token", "").encode(), self.token.encode()
        ):
            return web.json_response({"ok": False, "error": "bad token"}, status=403)

        part = q.get("part", "").strip().lower()
        description = self.part_descriptions.get(part)
        if not description:
            return web.json_response(
                {"ok": False, "error": f"unknown part: {part!r}"}, status=400
            )

        pressure = None
        if q.get("pressure"):
            try:
                pressure = int(q["pressure"])
            except ValueError:
                pressure = -1
            if not 0 <= pressure <= 100:
                return web.json_response(
                    {"ok": False, "error": "pressure must be 0-100"}, status=400
                )

        now = time.monotonic()
        remaining = self.cooldown - (now - self._last_touch.get(part, -1e9))
        if remaining > 0:
            return web.json_response(
                {"ok": False, "error": "cooldown", "retry_after": round(remaining, 1)},
                status=429,
            )

        text = description
        if pressure is not None and self.pressure_template:
            text += self.pressure_template.format(pressure=pressure)

        error = await self._inject(
            text, {"type": "touch", "part": part, "pressure": pressure}
        )
        if error:
            return web.json_response({"ok": False, "error": error}, status=503)
        self._last_touch[part] = now
        logger.info(
            f"[palmstar] Injected touch part={part} pressure={pressure}: {text}"
        )
        return web.json_response({"ok": True, "text": text})

    async def handle_voice(self, request: web.Request) -> web.Response:
        """Handle speech recorded by the ESP32 while the hand FSR is pressed.

        The request body is raw 16-bit little-endian mono PCM. It is saved as a
        WAV file, transcribed by an AstrBot speech-to-text provider, wrapped in
        voice_template and injected like a touch event.

        Query args:
            token: Shared secret, must match the configured token if set.
            rate: Sample rate of the PCM body, default 16000.

        Returns:
            JSON response. 200 injected, 400 bad body, 403 bad token,
            422 no speech recognized, 502 transcription failed,
            503 STT provider, platform or target not ready.
        """
        q = request.query
        if self.token and not hmac.compare_digest(
            q.get("token", "").encode(), self.token.encode()
        ):
            return web.json_response({"ok": False, "error": "bad token"}, status=403)

        try:
            rate = int(q.get("rate", 16000))
        except ValueError:
            rate = 0
        if not 8000 <= rate <= 48000:
            return web.json_response(
                {"ok": False, "error": "rate must be 8000-48000"}, status=400
            )
        pcm = await request.read()
        seconds = len(pcm) / (rate * 2)
        if seconds < 0.3:
            return web.json_response(
                {"ok": False, "error": "audio too short"}, status=400
            )

        # A configured provider id lets the doll use STT without turning on
        # AstrBot's global speech-to-text switch.
        if self.stt_provider_id:
            provider = self.context.get_provider_by_id(self.stt_provider_id)
        else:
            provider = self.context.get_using_stt_provider()
        if not isinstance(provider, STTProvider):
            logger.error(
                f"[palmstar] STT provider not available (id={self.stt_provider_id!r})"
            )
            return web.json_response(
                {"ok": False, "error": "stt provider not ready"}, status=503
            )

        wav_path = (
            Path(get_astrbot_temp_path())
            / f"palmstar_voice_{int(time.time() * 1000)}.wav"
        )
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm[: len(pcm) // 2 * 2])
            heard = (await provider.get_text(str(wav_path)) or "").strip()
        except Exception as e:
            logger.error(f"[palmstar] Speech transcription failed: {e}")
            return web.json_response(
                {"ok": False, "error": "transcription failed"}, status=502
            )
        finally:
            wav_path.unlink(missing_ok=True)

        if not heard:
            logger.info(f"[palmstar] No speech recognized in {seconds:.1f}s audio")
            return web.json_response(
                {"ok": False, "error": "no speech recognized"}, status=422
            )

        if "{text}" in self.voice_template:
            text = self.voice_template.replace("{text}", heard)
        else:
            text = self.voice_template + heard
        error = await self._inject(text, {"type": "voice", "seconds": seconds})
        if error:
            return web.json_response({"ok": False, "error": error}, status=503)
        logger.info(f"[palmstar] Injected voice ({seconds:.1f}s): {text}")
        return web.json_response({"ok": True, "text": text})

    async def _inject(self, text: str, marker: dict) -> str | None:
        """Inject text into the owner's QQ private chat as if the owner sent it.

        Args:
            text: Message text the bot will see.
            marker: Stored as event extra "palmstar" so hooks can recognize doll
                events (bypass_plugins, and later the reply-to-speech stage).

        Returns:
            None on success, otherwise a short error string.
        """
        if not self.target_qq.isdigit():
            logger.error("[palmstar] target_qq is not configured")
            return "target_qq not configured"

        # Find the OneBot (aiocqhttp) adapter the owner chats through.
        platform = None
        for p in self.context.platform_manager.platform_insts:
            meta = p.meta()
            if (self.platform_id and meta.id == self.platform_id) or (
                not self.platform_id and meta.name == "aiocqhttp"
            ):
                platform = p
                break
        if platform is None:
            logger.error(
                f"[palmstar] platform not found (platform_id={self.platform_id!r})"
            )
            return "platform not ready"

        # Bot QQ id and owner nickname, fetched once from OneBot and cached.
        if not self._self_id:
            try:
                bot = platform.get_client()
                info = await bot.get_login_info()
                self._self_id = str(info.get("user_id", ""))
                stranger = await bot.get_stranger_info(user_id=int(self.target_qq))
                self._nickname = str(stranger.get("nickname", ""))
            except Exception as e:
                logger.warning(f"[palmstar] Failed to query OneBot info: {e}")

        # Same shape the aiocqhttp adapter builds for a real private message,
        # so the event goes through the full pipeline and replies to this chat.
        abm = AstrBotMessage()
        abm.type = MessageType.FRIEND_MESSAGE
        abm.self_id = self._self_id
        abm.sender = MessageMember(self.target_qq, self._nickname or self.target_qq)
        abm.session_id = self.target_qq
        abm.message_id = f"palmstar_{int(time.time() * 1000)}"
        abm.message = [Plain(text)]
        abm.message_str = text
        abm.raw_message = None  # No OneBot event; sending falls back to session id.

        event = platform.create_event(abm)
        event.set_extra("palmstar", marker)
        platform.commit_event(event)
        return None
